#coding=utf-8
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import LoraConfig
import re

from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from src.data.odps_t2i_edit_data import preprocess_image
from inference.qwen_edit_think.demo_qwen_edit_think import vlm_prompt_thinking, load_thinker
from termcolor import colored
import argparse

class ImageEditor:
    def __init__(self, 
                 pretrained_model_path,
                 vlm_processor_path,
                 thinker_path=None,
                 train_ckpt_path=None,
                 use_lora=False,
                 device="cuda",
                 offload=True):
        self.device = device
        # Thinker (~16GB) + pipe (~57GB) together exceed a single 80GB card in bf16.
        # With offload=True the pipe is handed to diffusers' model cpu offload
        # (modules are swapped into VRAM on demand) while the Thinker stays resident:
        # peak VRAM ~= 16GB + 40GB DiT, peak RAM ~= 57GB.
        self.offload = offload

        # Load the Thinker first and move it onto the GPU immediately, freeing CPU
        # RAM before loading the much larger pipe; otherwise holding both on CPU at
        # once would need ~73GB of RAM.
        # NOTE: the Thinker must be loaded as a standalone model. pipe.text_encoder
        # is Qwen-Image-Edit's own vanilla Qwen2.5-VL, NOT the UnifiedThinker.
        thinker_path = thinker_path or vlm_processor_path
        print(f"Loading UnifiedThinker from {thinker_path}...")
        self.thinker = load_thinker(thinker_path, torch_dtype=torch.bfloat16)
        self.thinker = self.thinker.to(device)
        
        # Load the main model
        print("Loading image editing model...")
        self.pipe = QwenImageEditThinkPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # Optionally attach LoRA adapters
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
        
        # Load finetuned weights
        if train_ckpt_path:
            print(f"Loading finetuned weights from {train_ckpt_path}...")
            self._load_checkpoint(train_ckpt_path)

        # Load the VLM processor (used for the chain-of-thought)
        print("Loading VLM processor...")
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_processor_path)
        
        if self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)
        
        print("✓ Model loaded successfully!")
    
    def _load_checkpoint(self, ckpt_path):
        """Load checkpoint weights.

        Handles two naming conventions: thinker-editor training saves as
        dit./thinker., while older qwen-image-edit training saves as
        transformer./text_encoder.
        """
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
        
        # prefix -> target submodule
        targets = {
            "transformer.": self.pipe.transformer,
            "dit.": self.pipe.transformer,
            "text_encoder.": self.pipe.text_encoder,
            "thinker.": self.thinker,
        }
        
        matched = 0
        for prefix, module in targets.items():
            sub_state_dict = {
                k[len(prefix):]: v
                for k, v in final_state_dict.items()
                if k.startswith(prefix)
            }
            if not sub_state_dict:
                continue
            module.load_state_dict(sub_state_dict, strict=False)
            matched += len(sub_state_dict)
            print(f"Loaded {len(sub_state_dict)} params for '{prefix.rstrip('.')}'")
        
        # Fail loudly: if no prefix matches, the old code loaded nothing silently.
        if matched == 0:
            raise RuntimeError(
                f"No weights loaded from {ckpt_path}: none of the tensor names match the "
                f"expected prefixes {sorted(targets)}. Got e.g. "
                f"{sorted(final_state_dict)[:3]}"
            )
    

    def extract_answer_from_cot(self, prompt_cot, fallback=None):
        """Extract the <answer> tag content from the chain-of-thought.

        When no <answer> tag is found (e.g. the Thinker hit the token limit without
        closing the tag), fall back to `fallback` (the raw editing instruction)
        instead of using the whole CoT as the prompt -- the full reasoning text is
        not a usable edit prompt. Matches the rule in gen_risebench.py.
        """
        if prompt_cot is None:
            return fallback

        # Extract the content between <answer>...</answer> with a regex
        match = re.search(r'<answer>(.*?)</answer>', prompt_cot, re.DOTALL)

        if match:
            answer = match.group(1).strip()
            return answer
        else:
            # No tag found: fall back to the raw instruction
            print(colored(f"Warning: No <answer> tag found, falling back to raw instruction", "yellow"))
            return fallback
            
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
                self.thinker,
                max_new_tokens=4096
            )
            print(colored(f"Thinking: {prompt_cot}", "blue", attrs=["bold"]))
            prompt = self.extract_answer_from_cot(prompt_cot, fallback=prompt)
        print(colored(f"Final prompts: '{prompt}'", "green", attrs=["bold"]))
        # Prepare inputs
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
        
        # Run inference
        print(f"Editing image with prompt: '{prompt}'...")
        with torch.inference_mode():
            output = self.pipe(**inputs, height=height, width=width)
            result_image = output.images[0]
        
        # Save the result
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result_image.save(output_path)
        print(f"✓ Result saved to {output_path}")
        
        return result_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Image generator, e.g. model/Qwen-Image-Edit-2509")
    parser.add_argument("--processor_path", type=str, required=True,
                        help="Processor/tokenizer path, normally the UnifiedThinker checkpoint")
    parser.add_argument("--thinker_path", type=str, default=None,
                        help="UnifiedThinker-7B weights. Defaults to --processor_path.")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--no_offload", action="store_true",
                        help="Keep the whole pipeline on GPU instead of using "
                             "diffusers model-cpu-offload (needs ~75GB VRAM).")
    # Non-interactive single-shot mode: if --image and --prompt are given, run one
    # edit and exit; otherwise fall back to the interactive input() loop below.
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path. If set with --prompt, run one edit "
                             "non-interactively and exit.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Editing instruction for non-interactive mode.")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path for non-interactive mode.")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Skip the Thinker and feed the raw instruction (baseline).")
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="true_cfg_scale")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    editor = ImageEditor(
        pretrained_model_path=args.model_path,
        vlm_processor_path=args.processor_path,
        thinker_path=args.thinker_path,
        train_ckpt_path=args.ckpt_path if args.ckpt_path else None,
        use_lora=False,
        offload=not args.no_offload
    )
    
    
    print("\n" + "="*50)
    print("Image Editor Ready!")
    print("="*50 + "\n")

    # Non-interactive single-shot mode.
    if args.image and args.prompt:
        editor.edit_image(
            image_path=args.image,
            prompt=args.prompt,
            use_thinking=not args.no_thinking,
            output_path=args.output,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        print("\n✓ Editing completed!\n")
        return

    while True:
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        # Read the editing instruction
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
            # Run the edit
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
    