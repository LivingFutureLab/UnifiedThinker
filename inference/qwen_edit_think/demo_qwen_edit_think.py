#coding=utf-8
import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

from tqdm import tqdm
import json
import tempfile
from PIL import Image
import torch
from peft import LoraConfig
from transformers import AutoProcessor
from safetensors import safe_open
import textwrap

from src.data.odps_t2i_edit_data import preprocess_image
from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline

system_prompt_v3 = textwrap.dedent("""
    You are a **Visual-Language Model (VLM) Prompt Optimization Expert** specializing in image generation and editing. Your core task is to receive user instructions (potentially including a reference image), and after deep visual analysis and logical reasoning, output an **enhanced English prompt** (enhanced_prompt) for downstream Diffusion Models to generate high-quality images.

### **Three Core Principles (Guiding Principles)**

You must always adhere to the following three unshakeable principles, which are the foundation of all your actions.

1.  **Task Dichotomy**: Your primary judgment is to distinguish between **"Text-to-Image (T2I)"** and **"Image-to-Image (I2I)."**
    * **T2I is fundamentally about Creation**: Your `answer` must describe the entire scene in detail from scratch.
    * **I2I is fundamentally about Modification**: Your `answer` must be a precise instruction, describing **only the change** that needs to occur.

2.  **The "Golden Rule" for I2I (Modification Focus Principle)**: For any I2I task, your `answer` is **strictly forbidden from containing descriptions of any areas or elements that should remain unchanged.** The downstream model relies on the reference image to maintain constancy; restating these elements in the prompt will only lead to confusion and inconsistency.

3.  **The "Brain vs. Hand" Principle for Reasoning**: If the task requires logical reasoning, calculation, knowledge retrieval, or conceptual transformation, you must act as the **"Brain."**
    * Complete all thinking within the `<think>` tag and arrive at a **concrete, visual final result.**
    * In the `<answer>` tag, you must directly provide the **visual description of this result**, rather than asking the "Hand" (the downstream Diffusion Model) to repeat your thinking process.

### Guide for Thinking Process (<think> Tag Content)

You must structure your thinking within the `<think>` tag by naturally deconstructing the task through answering the following series of questions:

**Step 1: Input Analysis & Intent Identification**
-   **Basic Judgment**: Is this task "Text-to-Image" or "Image-to-Image"?
-   **Intent Verb**: What is the user's core intent? Is it **Add**, **Change**, **Replace**, **Isolate/Extract**, **Combine**, **Transform** (style/pose/concept), or **Solve/Draw** (solve and then draw)?

**Step 2: Reasoning Activation & Result Concretization**
-   **Reasoning Check**: Does fulfilling the intent from the previous step require reasoning beyond the literal meaning? (e.g., solving riddles, calculating, coordinate lookups, conceptual extension like "ten years later," or style imitation like "Picasso's style.")
-   **Execute Reasoning (If required)**: Immediately perform the required reasoning here.
-   **Result Statement**: After reasoning is complete, you must explicitly state: **"The concrete visual result of my reasoning is: [Write the specific, visual answer here]"**. Example: "...The concrete visual result of my reasoning is: the Sudoku grid with rows [1,3,4,2], [4,1,2,3]..."

**Step 3: Strategy Formulation & Prompt Construction**
-   **Comprehensive Decision**: Formulate the final `answer` based on the "Task Type" (T2I/I2I), the "User Intent Verb," and the "Concrete Reasoning Result" (if any).
-   **Principle-Based Construction**:
    * **If the task is "Text-to-Image"**:
        * **No Reasoning**: Freely enrich the details to build a complete scene description.
        * **With Reasoning**: Use the "Concrete Result" from Step 2 as the core subject and build the entire scene around it.
    * **If the task is "Image-to-Image"**:
        * Construct a clear, concise instruction sentence.
        * Must refer to the input image using phrases like "the given image" or "the input image." For multiple images, use placeholders `[image1]`, `[image2]`.
        * **Strictly adhere to the "Modification Focus Principle"** by describing only the change.
        * **If reasoning is involved**, the "change" itself is the "Concrete Result" obtained in Step 2.
        * *Good Instruction Examples*: `Add a large diamond ring to the thumb.`, `Replace the background with a lush green grassy field.`, `Isolate the sliced steak and place it on a solid white background.`, `Transform the image into the style of a Van Gogh painting.`, `Modify the image to show the final solved grid: top row is [1,3,4,2]...`

### Output Format (<answer> Tag Content)
Directly output a block of text, which must strictly adhere to the following format:
<think>
[Step 1: Input Analysis & Intent Identification] ...
[Step 2: Reasoning Activation & Result Concretization] ...
[Step 3: Strategy Formulation & Prompt Construction] ...
</think>

<answer>Enhanced English Prompt</answer>
""")


edit_cases = [

    {
        "ref_imgs_oss_path": [
            "/data/oss_bucket_1/jianchong.zq/benchmarks/RISEBench/data/temporal_reasoning_images/1.png"
        ],
        "prompt": "Draw what it will look like after being kept in a daily environment for a year.",
        "think": True
    },
]


def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
        # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
        # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'

def vlm_prompt_thinking(ref_images, edit_prompt, vlm_processor, pipe, max_new_tokens=128):
    content = []
    for i, im in enumerate(ref_images):
        content.append({"type": "text", "text": f"Input image {i+1}:\n"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": f"User instruction: {edit_prompt}"})
        
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt_v3}]
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Preparation for inference
    inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(pipe.text_encoder.device)

    with torch.no_grad():
        generated_ids = pipe.text_encoder.generate(
            **inputs, 
            max_new_tokens=max_new_tokens)
                    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    thinked_prompt = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(thinked_prompt) == 1, "batch is not 1"
    return thinked_prompt[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_model', type=str)
    parser.add_argument('--vlm_processor_path', type=str)
    
    parser.add_argument('--train_ckpt_file', type=str)
    parser.add_argument('--text_encoder_lora', action='store_true')
    parser.add_argument('--transformer_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=64)
        
    parser.add_argument('--und', action="store_true")
    parser.add_argument('--t2i', action="store_true")
    parser.add_argument('--edit', action="store_true")
    
    parser.add_argument('--pdb_debug', action='store_true')
    args = parser.parse_args()
    
    if args.pdb_debug:
        import pdb; pdb.set_trace()
    
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_processor_path)
    pipe = QwenImageEditThinkPipeline.from_pretrained(
        args.pretrained_model, 
        torch_dtype=torch.bfloat16)
    
    # 修改 edit system prompt
    pipe.prompt_template_encode = "<|im_start|>system\n" + EDIT_SYSTEM_PROMPT_20251117 + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    pipe.prompt_template_encode_start_idx = 2350

    if args.text_encoder_lora:
        text_encoder_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        pipe.text_encoder.language_model.add_adapter(text_encoder_lora_config)
    
    if args.transformer_lora:
        transformer_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipe.transformer.add_adapter(transformer_lora_config)
        
    if args.train_ckpt_file is not None and os.path.isdir(args.train_ckpt_file):
        # # 切片保存        
        index_path = os.path.join(args.train_ckpt_file, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        
        shard_files = set(weight_map.values())
        tensors_by_shard = {shard: [] for shard in shard_files}
        for tensor_name, shard_file in weight_map.items():
            tensors_by_shard[shard_file].append(tensor_name)
        
        final_state_dict = {}
        # 使用tqdm来显示进度条
        for shard_file, tensor_names in tqdm(tensors_by_shard.items(), desc="Loading shards"):
            shard_path = os.path.join(args.train_ckpt_file, shard_file)
            
            # 使用 safe_open，它不会立即加载所有张量到内存，而是创建一个映射
            # 这对于内存非常友好
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_name in tensor_names:
                    # 从分片文件中获取单个张量
                    tensor = f.get_tensor(tensor_name)
                    # 放入我们最终的 state_dict 中
                    final_state_dict[tensor_name] = tensor
        
        state_dict_transformer = {k.replace("transformer.", ""): v for k, v in final_state_dict.items() if k.startswith("transformer.")}
        state_dict_textencoder = {k.replace("text_encoder.", ""): v for k, v in final_state_dict.items() if k.startswith("text_encoder.")}
        assert len(final_state_dict) == len(state_dict_transformer) + len(state_dict_textencoder)
        
        if len(state_dict_transformer) > 0:
            missing_keys, unexpected_keys = pipe.transformer.load_state_dict(state_dict_transformer, strict=False)
            print("Load {} params for transformer, with {} unexpected_keys".format(len(state_dict_transformer), len(unexpected_keys)))
        if len(state_dict_textencoder) > 0:
            missing_keys, unexpected_keys = pipe.text_encoder.load_state_dict(state_dict_textencoder, strict=False)
            print("Load {} params for text_encoder, with {} unexpected_keys".format(len(state_dict_textencoder), len(unexpected_keys)))
        
    pipe = pipe.to("cuda")
        
    ## 理解任务
    if args.und:
        print("\n" + "-" * 20 + " 理解任务 " + "-" * 20)        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一位专业的视觉推理专家和图像编辑顾问。你的核心任务是：\n1.  **深入分析**: 接收用户提供的原始图片和编辑指令（edit prompt）。\n2.  **逻辑推理**: 结合图片内容（如物体材质、所处环境、当前状态）和生活常识、物理规律或特定艺术风格等，对编辑指令进行深度推理，预测出指令执行后最可能产生的视觉结果。\n3.  **精准描述**: 将推理出的编辑后图像样貌，用一段精炼、客观、富有画面感的文字描述出来。\n\n要求：\n- 直接返回最终的图像内容描述。\n- 描述内容控制在200字以内。\n- 禁止包含任何解释、分析过程或多余的客套话。"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/data/oss_bucket_1/jianchong.zq/benchmarks/RISEBench/data/temporal_reasoning_images/1.png",
                    },
                    {
                        "type": "text", 
                        "text": "Draw what it will look like after being kept in a daily environment for a year."
                    },
                ],
            }
        ]

        # Preparation for inference
        inputs = vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(pipe.text_encoder.device)

        # Inference: Generation of the output
        generated_ids = pipe.text_encoder.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
    
    ## T2I生成任务
    if args.t2i:
        print("\n" + "-" * 20 + " 图像生成任务 " + "-" * 20)
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }

        # Generate image
        prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197"'''
        negative_prompt = " " # using an empty string if you do not have specific concept to remove
        
        width, height = 1024, 1024
        #width, height = 1344, 736       # 16:9

        image = pipe(
            prompt=prompt + positive_magic[get_caption_language(prompt)],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_f:
            tmp_image_file = tmp_f.name     
        image.save(tmp_image_file)
        print(f"write to {tmp_image_file}")

    ## 图像编辑
    if args.edit:
        print("\n" + "-" * 20 + " 图像编辑任务 " + "-" * 20)
        for idx, data in enumerate(edit_cases):
            think = data.pop("think", False)
            
            prompt = data["prompt"]
            ref_imgs = [Image.open(f).convert("RGB") for f in data["ref_imgs_oss_path"]]
            ref_imgs  = [preprocess_image(im, max_area=1024*1024, adjust_ar=False) for im in ref_imgs]
            
            prompt_cot = None
            if think:
                prompt_cot = vlm_prompt_thinking(ref_imgs, prompt, vlm_processor, pipe, max_new_tokens=256)
                print("prompt_cot: {}".format(prompt_cot))
                try:
                    prompt_cot = json.loads(prompt_cot)['cot']
                except Exception as e:
                    prompt_cot = prompt

            width, height = ref_imgs[0].size
            print(f"width, height: {width} x {height}")
            
            # qwen-image-edit 使用固定 pixel area;
            # ours 和训练保持一致: 只对 > target_area 的图片进行下采样，不会对 < target_area 的图片进行上采样;
            if args.train_ckpt_file is None:
                fix_ref_img_pixel_area = True 
            else:
                fix_ref_img_pixel_area = False
            
            inputs = {
                "image": ref_imgs,
                "prompt": prompt,
                "prompt_cot": prompt_cot,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
                "fix_ref_img_pixel_area": fix_ref_img_pixel_area
            }
            with torch.inference_mode():
                output = pipe(**inputs, height=height, width=width)
                output_image = output.images[0]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_f:
                tmp_image_file = tmp_f.name     
            output_image.save(tmp_image_file)
            print(f"write to {tmp_image_file}")
