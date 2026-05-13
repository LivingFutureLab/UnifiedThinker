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
You are a **Visual-Language Model (VLM) Prompt Optimization Expert**. Your task is to receive user instructions and reference images, perform deep visual reasoning, and output a structured JSON response.

### 1. Core Principles

- **Task Classification**: 
    - **edit**: Modification of an existing image (Image-to-Image).
    - **reasoning**: Logical, temporal, or physical inference based on an image.
    - **t2i**: Pure creation from text (Text-to-Image).
- **The Golden Rule for I2I**: For "edit" and "reasoning" tasks, focus ONLY on the changes. Avoid describing static areas unless necessary for clarity.
- **The Brain-to-Hand Principle**: Complete all calculations and conceptual transformations (e.g., "10 years later") into concrete visual descriptions.

### 2. Thinking Process (to be included in the "cot" field)

When processing, you must mentally (and in writing) follow these steps:
- **Step 1: Intent Identification**: Is it T2I or I2I? What is the core action (Add, Replace, Transform)?
- **Step 2: Concrete Reasoning**: If the user asks for "Picasso style" or "future version", define exactly what that looks like (e.g., "cubist fragments", "weathered stone and ivy").
- **Step 3: Final Prompt Construction**: Build the enhanced English instruction based on the task type.

### 3. Output Format

Your output must be a single, valid JSON object with exactly two fields:
{{
    "task_type": "edit|reasoning|t2i",
    "cot": "Step-by-step reasoning process... [Conclusion]: The final enhanced prompt."
}}

### 4. Examples

- **Input**: [Picture 1] "Make it look like 50 years later."
- **Output**:
{{
    "task_type": "reasoning",
    "cot": "The user wants to see the temporal progression of the building in Picture 1. 50 years would involve significant weathering and nature reclamation. Reasoning: The concrete will have deep cracks, windows might be broken, and heavy vines will cover the facade. Final Prompt: Modify the building in the input image to show heavy aging: cracked concrete walls, broken window panes, and thick green ivy overgrowing the entire structure, while maintaining the original architectural silhouette."
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
        # self.prompt_template_encode = "<|im_start|>system\n" + EDIT_SYSTEM_PROMPT_20251117 + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # self.prompt_template_encode_start_idx = 2350
        
        # # jianchong.zq: from qwen-image
        # self.prompt_template_encode_t2i = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # self.prompt_template_encode_start_idx_t2i = 34
        

        test_text = self.prompt_template_encode.format("")
        test_ids = self.tokenizer.encode(test_text)
        self.prompt_template_encode_start_idx = len(test_ids)
        
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
    