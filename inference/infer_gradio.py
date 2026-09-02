#coding=utf-8
"""
Image editing Gradio demo
Suitable for recording demo videos
"""

import os
import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor
from peft import LoraConfig

from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from src.data.odps_t2i_edit_data import preprocess_image
from inference.qwen_edit_think.demo_qwen_edit_think import vlm_prompt_thinking, load_thinker
from termcolor import colored
import re

class ImageEditor:
    def __init__(self, 
                 pretrained_model_path,
                 vlm_processor_path,
                 thinker_path=None,
                 train_ckpt_path=None,
                 use_lora=False,
                 device="cuda",
                 offload=True):
        """Initialize the image editor."""
        self.device = device
        # Thinker (~16GB) + pipe (~57GB) together exceed a single 80GB card in bf16.
        # With offload=True the pipe uses diffusers' model cpu offload (modules swapped into VRAM on demand).
        self.offload = offload
        
        # Load the Thinker first and move it to GPU immediately, freeing CPU RAM before loading the larger pipe.
        # NOTE: pipe.text_encoder is Qwen-Image-Edit's own vanilla Qwen2.5-VL,
        # not the UnifiedThinker; the Thinker must be loaded separately.
        thinker_path = thinker_path or vlm_processor_path
        print(f"🧠 Loading UnifiedThinker from {thinker_path}...")
        self.thinker = load_thinker(thinker_path, torch_dtype=torch.bfloat16)
        self.thinker = self.thinker.to(device)
        
        print("🔄 Loading image editing model...")
        self.pipe = QwenImageEditThinkPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.bfloat16
        )
        
        if use_lora:
            print("🔧 Adding LoRA adapters...")
            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                lora_dropout=0,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.pipe.transformer.add_adapter(lora_config)
        
        if train_ckpt_path:
            print(f"📦 Loading finetuned weights from {train_ckpt_path}...")
            self._load_checkpoint(train_ckpt_path)
        
        print("🧠 Loading VLM processor...")
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_processor_path)
        
        if self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)
        
        print("✅ Model loaded successfully!")
    
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
            print(f"✓ Loaded {len(sub_state_dict)} params for '{prefix.rstrip('.')}'")
        
        # Fail loudly: if no prefix matches, the old code loaded nothing silently.
        if matched == 0:
            raise RuntimeError(
                f"No weights loaded from {ckpt_path}: none of the tensor names match the "
                f"expected prefixes {sorted(targets)}. Got e.g. "
                f"{sorted(final_state_dict)[:3]}"
            )

    def extract_answer_from_cot(self, prompt_cot):
        """Extract the <answer> tag content from the chain-of-thought."""
        if prompt_cot is None:
            return None
        
        # Extract the content between <answer>...</answer> with a regex
        match = re.search(r'<answer>(.*?)</answer>', prompt_cot, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            # If no tag is found, return the original content
            print(colored(f"Warning: No <answer> tag found, using full prompt", "yellow"))
            return prompt_cot

    def edit_image(self, 
                   image,
                   prompt,
                   use_thinking=True,
                   num_inference_steps=50,
                   guidance_scale=4.0,
                   seed=0,
                   progress=gr.Progress()):
        """
        Edit an image (Gradio version)
        """
        if image is None:
            return None, "", "❌ Please upload an image first!"
        
        if not prompt or not prompt.strip():
            return None, "", "❌ Please enter an editing instruction!"
        
        try:
            # Preprocess the image
            progress(0.1, desc="📸 Processing image...")
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            image = preprocess_image(image, max_area=1024*1024, adjust_ar=False)
            width, height = image.size
            
            # Generate the chain-of-thought
            prompt_cot = None
            if use_thinking:
                progress(0.2, desc="🧠 Generating thinking chain...")
                prompt_cot = vlm_prompt_thinking(
                    [image], 
                    prompt, 
                    self.vlm_processor, 
                    self.thinker,
                    # The Thinker's hierarchical CoT is long; too small a limit truncates it before <answer>,
                    # making extract_answer_from_cot fall back to using the whole CoT as the prompt, hurting quality.
                    max_new_tokens=4096
                )
                print(colored(f"Thinking: {prompt_cot}", "blue", attrs=["bold"]))
                prompt = self.extract_answer_from_cot(prompt_cot)
            print(colored(f"Final prompts: '{prompt}'", "green", attrs=["bold"]))
            # Prepare inputs
            progress(0.3, desc="🎨 Preparing for editing...")
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
            progress(0.4, desc="✨ Editing image...")
            with torch.inference_mode():
                output = self.pipe(**inputs, height=height, width=width)
                result_image = output.images[0]
            
            progress(1.0, desc="✅ Done!")
            
            info_text = (
                f"✅ **Editing Completed!**\n\n"
                f"📝 Prompt: {prompt}\n"
                f"🎲 Seed: {seed}\n"
                f"🔢 Steps: {num_inference_steps}\n"
                f"📊 Guidance Scale: {guidance_scale}\n"
                f"📐 Size: {width} x {height}"
            )
            
            return result_image, (prompt_cot if prompt_cot else "No thinking generated."), info_text
            
        except Exception as e:
            import traceback
            error_msg = f"❌ **Error occurred:**\n\n```\n{str(e)}\n{traceback.format_exc()}\n"
            return None, "", error_msg


def create_demo(editor):
    """Create the Gradio interface."""
    
    custom_css = """
    #main_title {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    #subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .output-image {
        max-height: 600px;
    }
    #thinker_box textarea {
        color: #1E90FF !important; /* dodger blue */
        font-family: 'Courier New', Courier, monospace;
        font-weight: 500;
        line-height: 1.5;
        background-color: #f0f8ff; /* light blue background */
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1 id='main_title'>🎨 UnifiedThinker Image Editor</h1>")
        gr.HTML("<p id='subtitle'>AI-Powered Image Editing with Thinking Chain</p>")
        
        with gr.Row():
            # Left: inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                prompt = gr.Textbox(
                    label="✏️ Editing Instruction",
                    placeholder="e.g., Make the sky blue and add clouds",
                    lines=3
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    use_thinking = gr.Checkbox(
                        label="🧠 Use Thinking Chain",
                        value=True,
                        info="Enable AI reasoning for better results"
                    )
                    
                    num_steps = gr.Slider(
                        label="🔢 Inference Steps",
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1
                    )
                    
                    guidance_scale = gr.Slider(
                        label="📊 Guidance Scale",
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5
                    )
                    
                    seed = gr.Number(
                        label="🎲 Random Seed",
                        value=0,
                        precision=0
                    )
                
                edit_btn = gr.Button("✨ Edit Image", variant="primary", size="lg")
            
            # Right: outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                output_image = gr.Image(
                    label="Edited Image",
                    type="pil",
                    height=400,
                    elem_classes=["output-image"]
                )
                
                # Put the Thinking content in its own Textbox to support long text and scrolling
                thinker_display = gr.Textbox(
                    label="🧠 Thinking Chain (Reasoning Process)",
                    placeholder="Thinking process will appear here...",
                    interactive=False,
                    lines=8,        # show 8 lines by default
                    max_lines=20,   # expand up to 20 lines, then scroll
                    elem_id="thinker_box"
                )
                
                info_box = gr.Markdown(
                    label="Info",
                    value="Click **Edit Image** to start"
                )
                
                with gr.Row():
                    download_btn = gr.File(label="📥 Download Result")
        
        # Event handling
        def edit_and_save(image, prompt, use_thinking, num_steps, guidance_scale, seed, progress=gr.Progress()):
            result_img, thinking, info = editor.edit_image(
                image, prompt, use_thinking, num_steps, guidance_scale, seed, progress
            )
            
            # Save the result
            if result_img:
                output_path = f"outputs/result_{seed}.png"
                os.makedirs("outputs", exist_ok=True)
                result_img.save(output_path)
                return result_img, thinking, info, output_path
            else:
                return None, thinking, info, None
        
        edit_btn.click(
            fn=edit_and_save,
            inputs=[input_image, prompt, use_thinking, num_steps, guidance_scale, seed],
            outputs=[output_image, thinker_display, info_box, download_btn]
        )
        
        gr.Markdown("""
        ---
        ### 📖 How to Use
        1. **Upload** an image you want to edit
        2. **Describe** what changes you want in natural language
        3. **Click** "Edit Image" and wait for the magic ✨
        4. **Download** your edited image
        """)
    
    return demo


import argparse

def main():
    """Main entry point."""
    # Parse command-line arguments with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/Qwen-Image-Edit-2509")
    parser.add_argument("--processor_path", type=str, default="model/UnifiedThinker-7B")
    parser.add_argument("--thinker_path", type=str, default=None,
                        help="UnifiedThinker-7B weights. Defaults to --processor_path.")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--no_offload", action="store_true",
                        help="Keep the whole pipeline on GPU instead of using "
                             "diffusers model-cpu-offload (needs ~75GB VRAM).")
    args = parser.parse_args()
    
    # Replace variables with the values from args
    PRETRAINED_MODEL = args.model_path
    VLM_PROCESSOR = args.processor_path
    TRAIN_CKPT = args.ckpt_path
    USE_LORA = False
    
    print("="*60)
    print("🚀 Initializing UnifiedThinker Image Editor")
    print("="*60)
    
    editor = ImageEditor(
        pretrained_model_path=PRETRAINED_MODEL,
        vlm_processor_path=VLM_PROCESSOR,
        thinker_path=args.thinker_path,
        train_ckpt_path=TRAIN_CKPT,
        use_lora=USE_LORA,
        offload=not args.no_offload
    )
    
    print("\n" + "="*60)
    print("✅ Creating Gradio Interface")
    print("="*60)
    
    demo = create_demo(editor)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )


if __name__ == "__main__":
    main()