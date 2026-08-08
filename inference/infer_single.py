#coding=utf-8
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import LoraConfig
import re

from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from src.data.odps_t2i_edit_data import preprocess_image
from inference.qwen_edit_think.demo_qwen_edit_think import vlm_prompt_thinking
from termcolor import colored
import argparse

class ImageEditor:
    def __init__(self, 
                 pretrained_model_path,
                 vlm_processor_path,
                 train_ckpt_path=None,
                 use_lora=False,
                 device="cuda"):
        self.device = device
        
        # 加载主模型
        print("Loading image editing model...")
        self.pipe = QwenImageEditThinkPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # 如果使用LoRA
        if use_lora:
            print("Adding LoRA adapters...")
            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                lora_dropout=0,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.pipe.transformer.add_adapter(lora_config)
        
        # 加载微调权重
        if train_ckpt_path:
            print(f"Loading finetuned weights from {train_ckpt_path}...")
            self._load_checkpoint(train_ckpt_path)
        
        self.pipe = self.pipe.to(device)
        
        # 加载VLM处理器（用于思维链）
        print("Loading VLM processor...")
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_processor_path)
        
        print("✓ Model loaded successfully!")
    
    def _load_checkpoint(self, ckpt_path):
        """加载检查点权重"""
        from safetensors import safe_open
        import json
        from tqdm import tqdm
        
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        shard_files = set(weight_map.values())
        tensors_by_shard = {shard: [] for shard in shard_files}
        
        for tensor_name, shard_file in weight_map.items():
            tensors_by_shard[shard_file].append(tensor_name)
        
        final_state_dict = {}
        for shard_file, tensor_names in tqdm(tensors_by_shard.items(), desc="Loading shards"):
            shard_path = os.path.join(ckpt_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_name in tensor_names:
                    final_state_dict[tensor_name] = f.get_tensor(tensor_name)
        
        # 分离transformer和text_encoder的权重
        state_dict_transformer = {
            k.replace("transformer.", ""): v 
            for k, v in final_state_dict.items() 
            if k.startswith("transformer.")
        }
        state_dict_textencoder = {
            k.replace("text_encoder.", ""): v 
            for k, v in final_state_dict.items() 
            if k.startswith("text_encoder.")
        }
        
        # 加载权重
        if state_dict_transformer:
            self.pipe.transformer.load_state_dict(state_dict_transformer, strict=False)
            print(f"Loaded {len(state_dict_transformer)} params for transformer")
        
        if state_dict_textencoder:
            self.pipe.text_encoder.load_state_dict(state_dict_textencoder, strict=False)
            print(f"Loaded {len(state_dict_textencoder)} params for text_encoder")
    

    def extract_answer_from_cot(self, prompt_cot):
        """从思维链中提取 <answer> 标签内容"""
        if prompt_cot is None:
            return None
        
        # 使用正则表达式提取 <answer>...</answer> 之间的内容
        match = re.search(r'<answer>(.*?)</answer>', prompt_cot, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            # 如果没有找到标签，返回原始内容
            print(colored(f"Warning: No <answer> tag found, using full prompt", "yellow"))
            return prompt_cot
            
    def edit_image(self, 
                   image_path,
                   prompt,
                   use_thinking=True,
                   output_path="output.png",
                   num_inference_steps=50,
                   guidance_scale=4.0,
                   seed=0):

        print(f"Loading image from {image_path}...")
        image = Image.open(image_path).convert("RGB")
        image = preprocess_image(image, max_area=1024*1024, adjust_ar=False)
        width, height = image.size
        
        prompt_cot = None
        if use_thinking:
            print("Generating thinking chain...")
            prompt_cot = vlm_prompt_thinking(
                [image], 
                prompt, 
                self.vlm_processor, 
                self.pipe,
                max_new_tokens=4096
            )
            print(colored(f"Thinking: {prompt_cot}", "blue", attrs=["bold"]))
            prompt = self.extract_answer_from_cot(prompt_cot)
        print(colored(f"Final prompts: '{prompt}'", "green", attrs=["bold"]))
        # 准备输入
        inputs = {
            "image": [image],
            "prompt": prompt,
            "prompt_cot": None,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": guidance_scale,
            "negative_prompt": " ",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "fix_ref_img_pixel_area": False
        }
        
        # 执行推理
        print(f"Editing image with prompt: '{prompt}'...")
        with torch.inference_mode():
            output = self.pipe(**inputs, height=height, width=width)
            result_image = output.images[0]
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result_image.save(output_path)
        print(f"✓ Result saved to {output_path}")
        
        return result_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    editor = ImageEditor(
        pretrained_model_path=args.model_path,
        vlm_processor_path=args.processor_path,
        train_ckpt_path=args.ckpt_path if args.ckpt_path else None,
        use_lora=False
    )
    
    
    print("\n" + "="*50)
    print("Image Editor Ready!")
    print("="*50 + "\n")
    
    while True:
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        # 输入编辑指令
        prompt = input("Enter editing instruction: ").strip()
        if not prompt:
            print("❌ Prompt cannot be empty")
            continue
        
        use_thinking = input("Use thinking chain? (y/n, default=y): ").strip().lower()
        use_thinking = use_thinking != 'n'
        
        output_path = input("Output path (default='output.png'): ").strip()
        if not output_path:
            output_path = "output.png"
        
        try:
            # 执行编辑
            result = editor.edit_image(
                image_path=image_path,
                prompt=prompt,
                use_thinking=use_thinking,
                output_path=output_path
            )
            print("\n✓ Editing completed!\n")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    main()
    