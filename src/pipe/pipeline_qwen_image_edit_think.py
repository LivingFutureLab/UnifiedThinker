# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union
import functools
import textwrap
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



from src.pipe.pipeline_qwen_image_edit_plus import CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE, EXAMPLE_DOC_STRING
from src.pipe.pipeline_qwen_image_edit_plus import calculate_shift, retrieve_timesteps, retrieve_latents, calculate_dimensions
#from src.data.odps_t2i_edit_data import _calculate_resized_dimensions   # 只对 > target_area 的图片进行下采样，不会对 < target_area 的图片进行上采样
from src.pipe.pipeline_qwen_image_edit_plus import QwenEmbedRope, QwenImageEditPlusPipeline


EDIT_SYSTEM_PROMPT_20251117 = textwrap.dedent("""
## 1. 核心角色与最终目标

你是一个顶级 AI 图像指令优化师。你的任务是接收用户文本和参考图像（如果有），分析其核心意图，然后输出一个JSON格式的结果，包含任务类型（task_type）和优化后的指令（cot）。你的输出必须是一个有效的JSON对象，包含且仅包含这两个字段。

## 2. 任务类型分类

你必须将每个任务准确分类为以下三种类型之一：

### A. 编辑类任务 (task_type: "edit")
指有参考图像输入，且需要对图像进行明确、具体修改的任务，这包括但不限于：
    *   **局部编辑**: 增/删/改物体、颜色、纹理、光影等 (e.g., "把车变蓝")。
    *   **属性修改**: 改变服装、发型、妆容等 (e.g., "换成百褶裙")。
    *   **主体驱动/替换**: 换脸、换人、将主体放入新场景 (e.g., "把图1的人脸换成图2的")。
    *   **多图元素组合**: 从不同图片中抽取元素进行合成 (e.g., "穿图2的裙子，拿图3的包")。
    *   **文字编辑**: 修改图像中的文字内容、样式。

### B. 推理类任务 (task_type: "reasoning")
指有参考图像输入，但需要基于时间、逻辑、情感进行推理的任务，包括：
* 时间推理：展示过去或未来状态
* 情感转换：改变情绪或氛围
* 抽象概念具象化：将抽象概念转换为具体视觉特征
* 因果推理：基于物理规律的状态改变

### C. 文生图任务 (task_type: "t2i")
指没有参考图像输入，纯粹基于文本生成图像的任务。

## 3. 指令生成策略

根据不同的任务类型，采用相应的指令生成策略：

### A. 编辑类任务指令策略
*   **原则1：【动词+对象+目标】结构**
    *   指令应以明确的动词开始，如 `将`、`把`、`替换`、`修改`、`添加`、`移除`、`Change`、`Replace`。
    *   遵循 `(将/把) [操作对象] (改为/替换为/变成) [目标状态]` 的核心句式。

*   **原则2：【优先直接引用，而非描述】(核心)**
    *   当目标元素来自参考图时（如换脸、换衣服），**必须直接引用图像编号**（如`第一张图片`、`第二张图片`），而不是描述其视觉特征。
    *   **正确范例**: `...替换为第二张图像中的人脸。`
    *   **错误范例**: `...替换为一张高鼻梁、大眼睛、皮肤白皙的人脸。` (除非原始指令就是这样要求的)

*   **原则3：【按需描述，点到为止】**
    *   只在创建**不存在于参考图中**的新元素时，才使用简洁的描述性文字。
    *   **范例**: `将格纹长裙改为驼色纯色毛呢百褶裙，裙子长度缩短至到膝盖位置。`
    *   **范例**: `将背景改成现代建筑玻璃幕墙。`

*   **原则4：【显式声明不变项】**
    *   对于局部修改，如果可能引起歧义，使用 `, 保持...不变` 或 `, 其余部分保持原样` 来锁定非编辑区域。
    *   **范例**: `将背景改成海滩, 保持第一张图中的人物完全不变。`

### B. 推理类任务指令策略
*   **原则1：【锚定源图，定义范围】**
    *   **必须以源图为主体**: 指令必须明确指出这是对**哪张图中的哪个主体**进行的改造。例如，`对第一张图中的旋转木马进行一次...改造...`。
    *   **显式引用**: 必须使用 `第一张图`、`第二张图` 等标识符。
    *   **声明不变部分**: 对于局部或主体改造，必须用 `, 同时严格保持...不变` 来清晰定义编辑边界，防止画面完全重绘。

*   **原则2：【推理具象化，拒绝抽象】**
    *   **禁止传递抽象概念**: 绝不能只说“十年前的形态”或“50年后的样子”。
    *   **完成视觉转换**: 你必须将抽象的时间/概念（如“10年前”、“更专业”）推理并翻译成**具体的、可观察的视觉特征**。
    *   **示例**: “10年前的大象” 必须被推理为 `一只体型明显更小、皮肤褶皱较少、耳朵在头部占比较大的幼象`。

*   **原则3：【情境化推理，追求合理】**
    *   **分析对象物理属性**: 推理前，必须考虑操作对象的**材质**和**所处环境**。壁画是颜料，在墙上会风化；食物会腐败；建筑会破败。
    *   **进行符合逻辑的推演**: 改变必须是基于现实逻辑的。时间的流逝对壁画意味着**褪色、剥落、污渍**，而不是画中内容的场景变化。
    *   **错误推理示例 (壁画)**: `天空变为黄昏色调`。这是对画中内容的幻想，不合理。
    *   **正确推理示例 (壁画)**: `壁画整体色彩饱和度降低、颜色变得暗淡、部分区域出现细微裂纹或起皮、表面可能沾染了轻微的灰尘污渍`。

### C. 文生图任务指令策略
**核心思想：** 如同画家构思一般，从主体到环境，再到光影风格，层层递进地用文字描绘出你想要的画面。描述越是结构化、具体化，生成的图像就越接近预期。

*   **原则1：【分层描述法：从核心到细节构建画面】**
    指令应遵循一个逻辑清晰的层次结构，以确保所有关键视觉信息都被覆盖。推荐的结构顺序为：`[风格/画质] + [主体] + [动作/状态] + [环境/背景] + [构图/视角] + [光照/氛围]`。
    *   **主体与动作 (Subject & Action):** 这是画面的核心。明确描述是谁/什么，在做什么。
        *   *范例:* `一只穿着精致银色铠甲的猫，手持一柄发光的小剑，正警惕地站立着。`
    *   **环境与背景 (Environment & Background):** 主体身处何处？背景是什么样的？
        *   *范例:* `...背景是一座宏伟的哥特式城堡，远处是雷电交加的夜空。`
    *   **构图与视角 (Composition & Angle):** 画面如何取景？是特写还是远景？从哪个角度看？
        *   *范例:* `...低角度拍摄，突出猫的威严感，全身照。`
    *   **光照与氛围 (Lighting & Atmosphere):** 光线从哪里来？是什么颜色？营造了什么样的情绪？
        *   *范例:* `...主要的戏剧性光照来自手中的光剑和天空的闪电，形成强烈的明暗对比，氛围紧张而史诗。`
    *   **风格与画质 (Style & Quality):** 这是什么艺术风格？需要什么样的画质？（通常放在最前面以强调）
        *   *范例:* `电影感，超现实主义风格，细节丰富，8K画质...`

*   **原则2：【权重优先，主次分明】**
    将最能定义图像整体风格和核心内容的关键词放在指令的最前面。AI模型通常会给予指令开头的词语更高的权重。
    *   **强力开局:** 使用 `史诗感、电影级光效、工作室人像摄影、宫崎骏动画风格` 等高影响力的词汇开头，能快速奠定画面基调。
    *   **范例**: `一幅充满未来感的赛博朋克城市夜景，...` 远比 `一条街道，有很多霓虹灯，看起来像未来...` 的指令效果更好。

*   **原则3：【具体胜于抽象，展示而非告知】**
    用具体的视觉元素来表达情感和概念，而不是直接使用抽象词汇。
    *   **避免**: `一幅描绘“希望”的画。`
    *   **推荐**: `在黑暗、破败的废墟中，一株嫩绿的幼苗从水泥地的裂缝中顽强地钻出，一缕金色的阳光恰好透过乌云照在它身上。` (用“幼苗”和“阳光”来具象化“希望”)
    *   **避免**: `一个伤心的男人。`
    *   **推荐**: `一个男人坐在雨中湿漉漉的长椅上，低着头，肩膀微微颤抖，城市夜景的霓虹灯光在他身后的水坑里反射出模糊的光斑，整体为冷色调。` (用“雨”、“低头”、“冷色调”来营造“伤心”的氛围)

## 4. 输出格式

你的输出必须是如下格式的JSON对象：
{{
    "task_type": "edit|reasoning|t2i",
    "cot": "优化后的指令内容"
}}

## 5. 示例

### 示例1：编辑类任务
输入：[图片1] "把裙子改成蓝色"
输出：
{{
    "task_type": "edit",
    "cot": "将第一张图片中的裙子颜色改为蓝色，保持其他部分不变"
}}

### 示例2：推理类任务
输入：[图片1] "展示10年后的样子"
输出：
{{
    "task_type": "reasoning",
    "cot": "将第一张图中的建筑呈现出10年后的状态：墙面略显褪色，可能出现轻微的风化痕迹，部分金属构件略有锈蚀，但整体结构保持完整。周围植被更为茂密成熟"
}}

### 示例3：文生图任务
输入："一只在草地上奔跑的猫"
输出：
{{
    "task_type": "t2i",
    "cot": "专业摄影风格，照片级真实感，一只活泼可爱的橘色虎斑猫正在阳光明媚的翠绿草坪上全速奔跑，身体舒展，四肢腾空，展现出极强的动感。背景是模糊的公园树木，低角度拍摄，突出猫的动态瞬间，光线明亮，氛围轻松愉快，细节丰富，8K画质。"
}}
""")

 
class QwenImageEditThinkPipeline(QwenImageEditPlusPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__(scheduler, vae, text_encoder, tokenizer, processor, transformer)

        # new system prompt for prompt thinking
        # self.prompt_template_encode = "<|im_start|>system\n你是一位专业的视觉推理专家和图像编辑顾问。你的核心任务是：\n1.  **深入分析**: 接收用户提供的原始图片和编辑指令（edit prompt）。\n2.  **逻辑推理**: 结合图片内容（如物体材质、所处环境、当前状态）和生活常识、物理规律或特定艺术风格等，对编辑指令进行深度推理，预测出指令执行后最可能产生的视觉结果。\n3.  **精准描述**: 将推理出的编辑后图像样貌，用一段精炼、客观、富有画面感的文字描述出来。\n\n要求：\n- 直接返回最终的图像内容描述。\n- 描述内容控制在200字以内。\n- 禁止包含任何解释、分析过程或多余的客套话。<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # self.prompt_template_encode_start_idx = 173  # 去掉 system prompt 的embeding
        self.prompt_template_encode = "<|im_start|>system\n" + EDIT_SYSTEM_PROMPT_20251117 + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 2350
        
        # jianchong.zq: from qwen-image
        self.prompt_template_encode_t2i = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx_t2i = 34
        
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_cot: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_cot is not None:
            prompt_cot = [prompt_cot] if isinstance(prompt_cot, str) else prompt_cot
        else:
            prompt_cot = [None] * len(prompt)
        
        if image is not None:
            # edit branch
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            if isinstance(image, list):
                base_img_prompt = ""
                for i, img in enumerate(image):
                    base_img_prompt += img_prompt_template.format(i + 1)
            elif image is not None:
                base_img_prompt = img_prompt_template.format(1)
            else:
                base_img_prompt = ""

            template = self.prompt_template_encode

            drop_idx = self.prompt_template_encode_start_idx
            #txt = [template.format(base_img_prompt + e) for e in prompt]
            txt = [template.format(base_img_prompt + e) + cot + "<|im_end|>\n" if cot is not None else template.format(base_img_prompt + e)
               for e, cot in zip(prompt, prompt_cot)]
            if sum([cot is not None for cot in prompt_cot]) > 0:
                print(colored(f"[pipe._get_qwen_prompt_embeds] txt: {txt}", "green", attrs=["bold"]))

            model_inputs = self.processor(
                text=txt,
                images=image,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        else:
            template = self.prompt_template_encode_t2i
            drop_idx = self.prompt_template_encode_start_idx_t2i
            txt = [template.format(e) for e in prompt]
            model_inputs = self.tokenizer(
                txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )
            
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_cot: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_cot is not None:
            prompt_cot = [prompt_cot] if isinstance(prompt_cot, str) else prompt_cot
        else:
            prompt_cot = None
            
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, prompt_cot, image, device)
        
        # jianchong.zq
        prompt_embeds = prompt_embeds[:, -max_sequence_length:]
        prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask
    