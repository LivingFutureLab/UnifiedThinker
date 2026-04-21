#coding=utf-8
"""
用于图像生成的CoT理解的数据格式:
{
    "ref_images": list of oss images, 编辑任务的参考图片。t2i 可以不用该字段。
    "prompt": 用户指令,
    "prompt_cot": 通用编辑任务可以不用该字段
    "task_type": 任务类型
}
"""

import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    
import io
import json
import random
import torch
from PIL import Image
import oss2
import tempfile
import traceback
from functools import partial
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from torch.utils.data import Dataset
from termcolor import colored

from src.data import utils
from src.data.odps_und_data import convert_to_qwen_vl_format, oss_download_file, CustomDataProcessor


system_prompt = """
# 角色与使命
你是一个顶级的AI视觉指令工程师 (You are a top-tier AI Visual Instruction Engineer)。你的核心任务是接收用户的输入（纯文本，或文本+参考图像），分析其意图，并按照指定的JSON格式输出结果。你的输出将用于驱动下游的AI视觉生成模型。

# 全局输出规则 (Global Output Rules)
1.  **JSON格式输出 (JSON Format Output)**: 你的输出**必须且只能是**一个严格遵循以下结构的JSON对象。
    ```json
    {
      "task_type": "string",
      "thinked_prompt": "string"
    }
    ```
2.  **task_type 定义**: `task_type` 字段的值必须是以下三者之一：
    *   `'general_edit'`: 用于明确、直接的图像编辑指令。
    *   `'reason_edit'`: 用于需要推理、联想或创意生成的抽象图像编辑指令。
    *   `'text_to_image'`: 用于所有文生图指令。
3.  **严禁额外内容 (No Extra Content)**: 严禁在JSON对象之外包含任何对话、解释、思考过程或Markdown格式（如`##`或`*`）。
4.  **语言一致 (Language Consistency)**: 在生成 `thinked_prompt` 的内容时，其语言必须与用户输入的语言保持一致（例如，用户输入中文，你输出中文；用户输入英文，你输出英文）。

# 核心工作流：判断与分派
你的工作流程基于一个核心判断：**用户是否提供了参考图像？** 你将根据此判断，在以下两种模式中选择一种来执行任务，并确定 `task_type` 的值。

---

## 模式一：图像编辑 (当用户提供参考图像时)
如果用户输入了文本和至少一张参考图像，你将扮演**图像指令分析师**的角色。你需先在内部判断用户请求属于以下哪一类，然后严格遵循对应策略生成JSON输出。

### 1. 核心策略：分类与执行
#### A. 常规执行类 (task_type: 'general_edit')
**定义**: 用户意图明确，操作具体。如：局部编辑（改颜色/物体）、换衣服、换脸、多图元素组合、修改文字。这类指令不需要复杂的推理或创意联想。

**生成策略**:
- 将 `task_type` 设置为 `'general_edit'`。
- 将 `thinked_prompt` 设置为**一个空字符串 `""`**。你不需要对这类指令进行重写或思考。

#### B. 推理创作类 (task_type: 'reason_edit')
**定义**: 用户意图模糊、抽象，或涉及基于逻辑、时间、情感的推演。如：“让他更开心”、“50年后的样子”、“增加节日氛围”。

**生成策略**:
- 将 `task_type` 设置为 `'reason_edit'`。
- **生成 `thinked_prompt`**: 你需要生成一段详细、具象化的描述性指令，并将其作为 `thinked_prompt` 的值。
    - **锚定主体**: 指令**必须**明确是`对第一张图中的[主体]`进行修改，并用`, 同时严格保持...不变`来限定编辑范围，防止画面被完全重绘。
    - **推理具象化**: **必须**将抽象概念（如“10年前”、“更专业”）翻译成**具体的、可观察的视觉特征**。**禁止**在最终指令中出现抽象词汇。
        - **例**: “10年前的大象” -> 推理为 `一只体型明显更小、皮肤褶皱较少、耳朵占比较大的幼象`。
    - **逻辑合理性**: 你的推理必须符合物理和现实逻辑。时间的流逝对壁画意味着**褪色、剥落、污渍**，而不是画中内容的场景变化。

### 2. 工作范例 (模式一)

#### 范例1: 常规执行 ('general_edit')
- **用户输入**: "让图1的模特，穿上图2的裙子和图3的鞋子，背景不变"
- **你的输出 (JSON)**:
{
  "task_type": "general_edit",
  "thinked_prompt": ""
}

#### 范例2: 推理创作 ('reason_edit')
- **用户输入**: "画出它十年前的样子。" (图为一只成年大象)
- **你的输出 (JSON)**:
{
  "task_type": "reason_edit",
  "thinked_prompt": "将第一张图中的成年大象重绘为它十年前的幼年形态。具体来说，生成一只体型明显更小、皮肤褶皱较少、耳朵在头部占比较大的幼象，姿态和朝向与原图保持一致。同时，严格保持背景的稀树草原环境、光照和构图完全不变。"
}

---

## 模式二：文生图 (task_type: 'text_to_image')
如果用户只输入了文本，你将扮演**文生图指令重写专家**的角色。你的目标是将简单的文本转化为一个高度详细、结构化、能引导AI生成高“对齐度”和高“美学”得分图像的指令。

**生成策略**:
- 将 `task_type` 设置为 `'text_to_image'`。
- **生成 `thinked_prompt`**: 你需要遵循以下原则，用**目标语言（与用户输入一致）**构建一个高质量的指令，并将其作为 `thinked_prompt` 的值。

### Part 1: Maximizing 'Alignment' Score (Content & Detail)
(这部分用于构建 `thinked_prompt` 的内容)
1.  **Core Structure - Deconstruct and Expand**:
    *   识别并丰富核心的**主体 (Subject)**、**场景 (Setting)**和**动作 (Action)**。详细描述主体的外观、衣着、表情；用前景、背景和环境元素来细化场景。
    *   添加大量次要细节和物体，构建一个丰富、无歧义的画面。
2.  **Composition and Spatial Logic**:
    *   明确数量和位置。使用如“三只呈V字形飞行的鸟”、“一个放在热气腾腾的咖啡杯*左边*的书”、“一只睡在*一叠报纸上*的猫”等短语。
    *   清晰定义互动关系：“一个骑士用闪光盾牌格挡巨龙的火焰吐息”。
3.  **Emotional and Atmospheric Nuance**:
    *   不要只说出情感，而是将其转化为具体的视觉语言。
    *   不要用“悲伤的肖像”，而是描述“一幅带有忧郁情绪的肖像，来自单扇窗户的柔和暗淡光线，由蓝色和灰色构成的柔和色调，主体的目光朝下”。
4.  **Stylistic Depth**:
    *   不要只命名风格，要补充其关键的技术和美学特征。
    *   不要用“梵高风格”，而是写“后印象主义风格，具有厚重、富有表现力的厚涂笔触、旋转的天空和充满活力的情感调色板”。
    *   不要用“浮世绘风格”，而是写“浮世绘风格，以平面色块、粗黑轮廓和不对称构图为特点”。
5.  **Entity Specificity and Context**:
    *   如果指令提到特定的人物、角色或地标，确保其所处的环境同样详细。
    *   不要用“阿尔伯特·爱因斯坦”，而是写“阿尔伯特·爱因斯坦在他1940年代的普林斯顿办公室里，站在一块写满物理方程式的黑板旁，桌上散落着书籍和文件”。
6.  **Imaginative Coherence**:
    *   对于幻想或超现实的物体，描述不同组件如何融合。使用“无缝融合”、“有机地过渡到”、“由一整块...构成”等短语。
    *   描述“用水做的时钟”：“一个时钟，指针是不断流动的水流，被一个维持其形状的闪亮无形力场包裹，数字由冒泡的空气形成”。
7.  **Precise Text Rendering**:
    *   如果需要文字，使用格式：`text "你想要的文字"`。
    *   关键是描述其外观和融合方式：“文字'OPEN'用发光的红色霓虹灯字母写在一个质朴的木牌上，木牌挂在夜间一栋砖砌建筑正面的生锈链条上。霓虹灯在后面的墙上投下柔和的红光。”

### Part 2: Maximizing 'Aesthetic' Score (Quality & Realism)
(这部分用于构建 `thinked_prompt` 的内容)
*   在构建完详细描述后，**必须**在指令末尾附加一组“质量关键词”，以追求技术上的卓越。
*   **示例关键词组**: `, masterpiece, best quality, ultra-detailed, hyperrealistic, photorealistic, sharp focus, 8k resolution, cinematic lighting, professional photography, intricate details, physically-based rendering, anatomically correct`。

### 工作范例 (模式二)
*   **用户输入**: `a cat`
*   **你的输出 (JSON)**:
{
  "task_type": "text_to_image",
  "thinked_prompt": "A close-up photograph of a fluffy calico cat with bright green eyes, peacefully napping on a sun-drenched windowsill. Outside the window, a blurry background of green garden trees is visible. The cat's fur is incredibly detailed, showing individual strands and soft textures. The scene has a warm, serene atmosphere with soft, natural morning light casting gentle shadows. masterpiece, best quality, ultra-detailed, photorealistic, sharp focus, 8k resolution, cinematic lighting, professional photography, intricate details."
}
"""

Image.MAX_IMAGE_PIXELS = 20_000_000

def read_json_data(file_name):
    if file_name.endswith(".json"):
        with open(file_name, 'r') as f:
            datas = json.load(f)
        return datas
    elif file_name.endswith(".jsonl"):
        datas = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "": continue
                datas.append(json.loads(line))
        return datas
    else:
        raise ValueError("not supported file: {}".format(file_name))
    
class LocalThinkerDataset(Dataset):
    def __init__(
        self, 
        data_files, 
        data_weights=None,
        qwenvl_pretrained: str = "/data/oss_bucket_0/jianchong.zq/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct",
        max_sequent_length: int = 4096,
        oss_access_id="",
        oss_access_key="",
        bucket_name="",
        oss_endpoint="",
    ):
        super(LocalThinkerDataset, self).__init__()
        self.bucket_name = bucket_name
        self.oss_bucket = oss2.Bucket(oss2.Auth(oss_access_id, oss_access_key), oss_endpoint, bucket_name)
        
        self.datas = []    
        if isinstance(data_files, str):
            data_files = [data_files]
        
        if data_weights is None:
            data_weights = [1.0] * len(data_files)
        else:
            assert len(data_weights) == len(data_files)
            
        for data_file, data_weight in zip(data_files, data_weights):
            tmp_datas = None
            if not os.path.exists(data_file):
                # download from oss 
                object_key = data_file.replace(f"oss://{self.bucket_name}/", "")
                suffix = os.path.splitext(object_key)[1]
                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_f:
                    if oss_download_file(self.oss_bucket, object_key, tmp_f.name):
                        tmp_datas = read_json_data(tmp_f.name)
            else:
                tmp_datas = read_json_data(data_file)
            
            if tmp_datas is not None:
                tmp_datas = tmp_datas * int(data_weight)    # 重采样
                print(colored("[Dataset] {} samples read from: {}".format(len(tmp_datas), data_file), "green", attrs=["bold"]))
                
                self.datas += tmp_datas
                
        print(colored("[Dataset] total {} samples".format(len(self.datas)), "green", attrs=["bold"]))
        
        self.vlm_processor = AutoProcessor.from_pretrained(
            qwenvl_pretrained,
            min_pixels=64 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )                
        self.max_sequent_length = max_sequent_length
        
        # Pre-tokenize templates to find assistant responses
        # Note: The exact string depends on the chat template. For Qwen, it includes a newline.
        self.assistant_prompt_ids = self.vlm_processor.tokenizer.encode(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )
        self.im_end_id = self.vlm_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")        
        self.collate_fn = partial(CustomDataProcessor.collate_fn, processor=self.vlm_processor)
                
        random.seed(0)
        random.shuffle(self.datas)
    
    def __len__(self):
        return len(self.datas)
    
    def _download_and_open_image(self, oss_path: str) -> Optional[Image.Image]:
        """Downloads an image from OSS and returns it as a PIL Image object."""
        try:
            object_key = oss_path.replace(f"oss://{self.bucket_name}/", "")
            suffix = os.path.splitext(object_key)[1]
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_f:
                if oss_download_file(self.oss_bucket, object_key, tmp_f.name):
                    image = Image.open(tmp_f.name).convert("RGB")
                    return image
                else:
                    return None
        except Exception as e:
            utils.logger.warning(f"Failed to download or open image {oss_path}: {e}")
            utils.logger.info("Try load from /data/oss_bucket_0/")
            image_path = oss_path.replace(f"oss://{self.bucket_name}/", "/data/oss_bucket_0/")
            image = Image.open(image_path).convert("RGB")
            return image
                    
    def __getitem__(self, idx):
        while True:
            data = self.datas[idx]
            try:       
                record_id = data.get("record_id", f"data_{idx}")
                ref_images = data.get("ref_images", [])     # 参考图像
                prompt = data['prompt']                     # 用户输入指令
                prompt_cot = data.get("prompt_cot", "")     # 思考后的指令, 对于通用编辑任务，prompt_cot 为空
                task_type = data['task_type']
                
                assert isinstance(ref_images, (tuple, list)), "ref_images must be list"
                assert isinstance(prompt, str)
                assert isinstance(prompt_cot, str)
                assert isinstance(task_type, str)
                assert task_type in ['general_edit', 'reason_edit', 'text_to_image'], f"task_type of {task_type} is not supported."
                
                # if task_type == 'general_edit' and prompt_cot != "":
                #     prompt_cot = ""
                #     print("prompt_cot set to empty for general_edit")
                
                content = []
                for i, im in enumerate(ref_images):
                    im = self._download_and_open_image(im)
                    content.append({"type": "text", "text": f"Input image {i+1}:\n"})
                    content.append({"type": "image", "image": im})
                content.append({"type": "text", "text": f"User instruction: {prompt}"})
                
                # 动态调整 max_pixels
                if len(ref_images) > 0:
                    self.vlm_processor.image_processor.max_pixels = max(1280 // len(ref_images), 256) * 28 * 28
                else:
                    self.vlm_processor.image_processor.max_pixels = 1280 * 28 * 28
                
                answer = {
                    "task_type": task_type,
                    "thinked_prompt": prompt_cot
                }
                
                conversations = [
                    { "role": "system", "content": system_prompt},
                    { "role": "user", "content": content},
                    { "role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
                ]
                
                text = self.vlm_processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(conversations)

                inputs = self.vlm_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=False,       # Padding will be handled by the collate_fn
                    max_length=self.max_sequent_length, 
                    truncation=True,    # 当序列超过最大长度时进行截断
                )      
                input_ids = inputs["input_ids"][0]
                # Create labels tensor: mask out everything except assistant's responses
                labels = torch.full_like(input_ids, -100)
                
                # Find all occurrences of the assistant prompt
                input_ids_list = input_ids.tolist()
                assistant_prompt_len = len(self.assistant_prompt_ids)
                
                i = 0
                while i < len(input_ids_list):
                    # Check if the sequence from index `i` matches the assistant prompt
                    if input_ids_list[i:i + assistant_prompt_len] == self.assistant_prompt_ids:
                        # Found an assistant response start
                        response_start_idx = i + assistant_prompt_len
                        # Find the end of the response, marked by <|im_end|>
                        try:
                            response_end_idx = input_ids_list.index(self.im_end_id, response_start_idx)
                        except ValueError:
                            # If <|im_end|> is not found, assume the response goes to the end
                            response_end_idx = len(input_ids_list)
                        
                        # Copy the token IDs of the response to the labels tensor
                        labels[response_start_idx:response_end_idx] = input_ids[response_start_idx:response_end_idx]
                        
                        # Move index `i` past this found response
                        i = response_end_idx
                    else:
                        i += 1

                # Also mask out special visual tokens in labels, although they are likely
                # already -100 because they are not part of an assistant response.
                visual_tokens_ids = [
                    self.vlm_processor.tokenizer.convert_tokens_to_ids(self.vlm_processor.image_token),
                    self.vlm_processor.tokenizer.convert_tokens_to_ids(self.vlm_processor.video_token)
                ]
                for token_id in visual_tokens_ids:
                    if token_id is not None:
                        labels[labels == token_id] = -100
                
                # Add the carefully crafted labels to the output
                inputs['labels'] = labels.unsqueeze(0) # Add batch dimension
                
                # Squeeze out the batch dimension added by the vlm_processor
                for k in inputs:
                    if inputs[k] is not None and isinstance(inputs[k], torch.Tensor):
                        if k not in ["pixel_values", "image_grid_thw"]: # These are already batched correctly
                            inputs[k] = inputs[k].squeeze(0)     
                            
                if idx % 100 == 0:
                    if "pixel_values" in inputs:
                        print(f"[Dataset-und] [record_id]: {record_id}, [pixel_values]: {inputs['pixel_values'].shape}, [text]: {repr(text)}")  
                    else:
                        print(f"[Dataset-und] [record_id]: {record_id}, [text]: {repr(text)}")  
                                    
                inputs["record_id"] = record_id
                return inputs
                
            except Exception as e:
                utils.logger.error(
                    f"Error processing sample: {str(e)}\n{traceback.format_exc()}"
                )
                idx = random.randint(0, len(self.datas) - 1)


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    train_dataset = LocalThinkerDataset(
        data_files=["/data/oss_bucket_1/jianchong.zq/json_file_4_train/reason_edit_cot_for_thinker_training_72k.jsonl"],
        data_weights=[1.0],
        qwenvl_pretrained="/data/oss_bucket_0/jianchong.zq/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct",
        max_sequent_length=16384,
    )
    
    utils.logger.info("Dataset preparation completed")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn)
    utils.logger.info("DataLoader preparation completed")
    
    for i, data_dict in enumerate(dataloader):
        utils.logger.info(f"Processing batch {i}")
        if i > 100:
            break
    