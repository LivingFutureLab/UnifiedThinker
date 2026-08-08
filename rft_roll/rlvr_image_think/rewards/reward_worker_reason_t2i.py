#coding=utf-8
import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import math
import textwrap
from typing import Optional, Union, Dict, List, Any
import json
import random
import demjson3
#import re
import regex as re  # pip install regex
import torch
import requests
import time
import traceback
import numpy as np
from functools import partial
import tensordict
from PIL import Image
from tensordict import TensorDict
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from ray.util.timer import _Timer

from roll.configs.worker_config import WorkerConfig, StrategyArguments, ModelArguments
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.platforms import current_platform
from roll.utils.logging import get_logger
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *

from rlvr_image_think.image_edit_think_pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from rlvr_image_think.image_edit_think_pipe.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from rlvr_image_think.image_edit_think_pipe.util import preprocess_image
from rlvr_image_think.utils import download_model_mos


# 参考 https://github.com/PKU-YuanGroup/WISE/blob/main/gpt_eval.py

system_prompt = """Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**  
- PROMPT: [User's original prompt to]  
- EXPLANATION: [Further explanation of the original prompt] 
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

--- 
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{}"
EXPLANATION: "{}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""


def custom_image_edit_model_provider(tokenizer, model_args, training_args=None, is_trainable=False, is_think_model=True):  
    model_args.model_name_or_path = download_model_mos(model_args.model_name_or_path)  
    if is_think_model:      
        # 使用 think version image-edit
        model = QwenImageEditThinkPipeline.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=torch.bfloat16)  
    else:
        # 使用 non-think version image-edit, 直接将 cot 作为新的 prompt 使用
        model = QwenImageEditPlusPipeline.from_pretrained(model_args.model_name_or_path, 
                                        torch_dtype=torch.bfloat16)  
    return model

def custom_vlm_model_provider(tokenizer, model_args, training_args=None, is_trainable=False):
    """
    Loads the VLM model, tokenizer, and processor.
    """
    if "qwen2.5-vl" in model_args.model_name_or_path.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "qwen3-vl" in model_args.model_name_or_path.lower():
        if "qwen3-vl-8b" in model_args.model_name_or_path.lower():
            model_class = Qwen3VLForConditionalGeneration
        elif "qwen3-vl-30b-a3b" in model_args.model_name_or_path.lower():
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"not supported model: {model_args.model_name_or_path}")
    else:
        raise ValueError()
        
    model_args.model_name_or_path = download_model_mos(model_args.model_name_or_path)   
    model = model_class.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
        
    model.processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )    
    get_logger().info(f"Successfully loaded VLM model, tokenizer, and attached the processor.")    
    return model

def qwen_vl_generate(model, messages):
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Process inputs
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate output
    generation_config = {
        "max_new_tokens": 512,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.1,
        "top_p": None,
    }
    generated_ids = model.generate(**inputs, **generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""

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

def extract_scores(txt: str) -> Dict[str, float]:
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
    matches = re.findall(pat, txt, re.IGNORECASE)
    out = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    return out
   
class ReasonT2IRewardWorker(Worker):
    """
    Reward Worker that uses two VLMs (Editor and Judge) to compute image editing rewards.
    """
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size

        self.edit_strategy: Optional[InferenceStrategy] = None
        self.vlm_strategy: Optional[InferenceStrategy] = None
        self.debug_mode = False
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        
        # 从 worker_config 中获取 clusters 列表，这个列表由 YAML 文件定义
        cluster_configs = self.worker_config.clusters
        if not cluster_configs or len(cluster_configs) < 2:
            raise ValueError(
                "Configuration error: Expected 'clusters' list with at least 2 entries in the YAML file for ReasonT2IRewardWorkerV2."
            )
        
        # 从 worker_config 中获取自定义的，reward 相关的超参数, 具体配置见 'rft_roll/rlvr_image_think/rlvr_image_think_config.py'
        self.edit_max_area = self.worker_config.edit_max_area
        self.edit_sampling_steps = self.worker_config.edit_sampling_steps
        self.edit_is_think_model = self.worker_config.edit_is_think_model   # 是否使用 qwen-edit-think 模型，还是使用 qwen-edit
        self.logger.info(f"Edit sampling set edit_max_area to {self.edit_max_area}, edit_sampling_steps to {self.edit_sampling_steps}, edit_is_think_model to {self.edit_is_think_model}")

        self.is_offload_state_edit_model = self.worker_config.is_offload_state_edit_model 
        self.is_offload_state_vlm = self.worker_config.is_offload_state_vlm
        self.logger.info(f"Setting is_offload_state_edit_model to {self.is_offload_state_edit_model}, is_offload_state_vlm to {self.is_offload_state_vlm}")
        
        # image-edit-model
        self.logger.info("Initializing editor strategy from cluster[0]...")
        # 在调用 create_strategy 之前，手动将第一个 cluster 的配置赋给 worker_config 的顶层属性, 这样 create_strategy 就能找到它需要的配置
        self.worker_config.model_args = ModelArguments(**cluster_configs[0].get("model_args", {}))
        self.worker_config.strategy_args = StrategyArguments(**cluster_configs[0].get("strategy_args", {}))
        self.edit_strategy = create_strategy(worker=self)  
        #self.edit_strategy.initialize(model_provider=custom_image_edit_model_provider)  
        self.logger.info(f"Manually loading model from provider: custom_image_edit_model_provider")
        edit_model = custom_image_edit_model_provider(tokenizer=None, model_args=self.worker_config.model_args, is_think_model=self.edit_is_think_model)
        self.edit_strategy.model = edit_model
        if self.is_offload_state_edit_model:
            self.edit_strategy.offload_states()     # 卸载状态以节省内存  
        
        # vlm-model
        self.logger.info("Initializing VLM strategy from cluster[1]...")
        # 使用第二个 cluster 的配置
        self.worker_config.model_args = ModelArguments(**cluster_configs[1].get("model_args", {}))
        self.worker_config.strategy_args = StrategyArguments(**cluster_configs[1].get("strategy_args", {}))
        self.vlm_strategy = create_strategy(worker=self)  
        self.vlm_strategy.initialize(model_provider=custom_vlm_model_provider)  
        if self.is_offload_state_vlm:
            self.vlm_strategy.offload_states()     # 卸载状态以节省内存  
        
        # 清理临时的顶层属性，避免后续逻辑混淆
        self.worker_config.model_args = None
        self.worker_config.strategy_args = None
        
        current_platform.init()

        if pipeline_config.system_envs.get("RAY_DEBUG", "") == "legacy":
            self.debug_mode = True
            self.logger.info("进入 Debug 模式")
                
    @torch.no_grad()
    def _generate_t2i_images(self, data: DataProto):  
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            #breakpoint()
            self.logger.info("进入 Debug 模式: _generate_t2i_images")
            
        max_area = self.edit_max_area 
        num_inference_steps = self.edit_sampling_steps
        edited_images = []
        
        # 直接获取 strategy 管理的模型实例
        edit_model = self.edit_strategy.unwrap_model() # 或者直接用 self.edit_strategy.model
        
        pred_prompt_cot_list = edit_model.processor.batch_decode(data.batch["responses"], skip_special_tokens=True)
        
        for prompt, prompt_cot, domain in zip(
            data.non_tensor_batch["edit_prompt"], pred_prompt_cot_list, data.non_tensor_batch['domain']
        ):   
            assert "t2i" in domain, "wrong data domain, this reward worker is used for t2i task."

            width, height = 1024, 1024  
            
            # For prediction from format:  <think> xxxx </think><answer> xxx </answer>
            match = re.search(r'<answer>(.*?)</answer>', prompt_cot, re.DOTALL)
            if match:
                self.logger.info("[ReasonT2IRewardWorker] FUll prompt_cot: {}".format(repr(prompt_cot)))
                prompt_cot = match.group(1).strip()
                self.logger.info("[ReasonT2IRewardWorker] Answer of prompt_cot: {}".format(repr(prompt_cot))) 
            
            # positive_magic = {
            #     "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            #     "zh": ", 超清，4K，电影级构图." # for chinese prompt
            # }
            # prompt_cot = prompt_cot + positive_magic[get_caption_language(prompt_cot)]
            
            if self.edit_is_think_model:
                inputs = {
                        "prompt": prompt,
                        "prompt_cot": prompt_cot,
                        "generator": torch.manual_seed(0),
                        "true_cfg_scale": 4.0,
                        "negative_prompt": " ",
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1,
                        "fix_ref_img_pixel_area": False
                    }
                self.logger.info(f"[ReasonT2IRewardWorker] [qwen-image-edit-think] - generate image with prompt: {prompt};  prompt_cot: {repr(prompt_cot)}")
            else:
                inputs = {
                        "prompt": prompt_cot,
                        "generator": torch.manual_seed(0),
                        "true_cfg_scale": 4.0,
                        "negative_prompt": " ",
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1,
                        "fix_ref_img_pixel_area": False
                    }
                self.logger.info(f"[ReasonT2IRewardWorker] [qwen-image-edit] (origin prompt: {prompt})- generate image with prompt_cot: {repr(prompt_cot)}")
                
            output = edit_model(**inputs, height=height, width=width)
            output_image = output.images[0]
            edited_images.append(output_image)
        return edited_images
    
    @torch.no_grad()
    def _score_t2i_images(self, edited_images, data: DataProto):
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            #breakpoint()
            self.logger.info("进入 Debug 模式: _score_t2i_images")
            
        vlm_model = self.vlm_strategy.unwrap_model() # 或者 self.vlm_strategy.model
        
        scores = []
        for i, (image, prompt, domain, prompt_cot) in enumerate(zip(
            edited_images, data.non_tensor_batch["edit_prompt"], data.non_tensor_batch["domain"], data.non_tensor_batch["edit_prompt_cot"]
        )):
            try:
                assert "t2i" in domain, "wrong data domain, this reward worker is used for t2i task."
                assert isinstance(image, Image.Image), "post_edit_image is not Image"
                
                image = image.convert("RGB")
                final_prompt = system_prompt.format(prompt, prompt_cot)
                
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": final_prompt},
                                    {"type": "text", "text": "Generated image:\n"},
                                    {"type": "image", "image": image}
                                    ]
                    }]
                output_text = qwen_vl_generate(vlm_model, messages)
                output_score = extract_scores(output_text)
                self.logger.info("[ReasonT2IRewardWorker] [_score_t2i_images] output_score: {}".format(output_score))
                
                Consistency = output_score.get("consistency", 0)
                Realism = output_score.get("realism", 0)
                Aesthetic = output_score.get("aesthetic_quality", 0)
                final_score = (Consistency + Realism + Aesthetic) / 3.0
            except Exception as e:
                print("error of {} during _scroe_t2i_images".format(str(e)))
                final_score = 0.0
            scores.append(final_score)
                
        return scores
        
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)  
    def compute_rewards(self, data: DataProto):  
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            #breakpoint()
            self.logger.info("进入 Debug 模式")
            
        global_step = data.meta_info.get("global_step", 0)
        metrics = {}  
        is_offload_states = data.meta_info.get("is_offload_states", True)  
        
        # is_offload_edit = getattr(self.edit_strategy.args, 'offload_states', True)
        # is_offload_vlm = getattr(self.vlm_strategy.args, 'offload_states', True)
        
        time_0 = time.time()
        # 阶段1: 使用 Image Edit Model 生成编辑后的图片  
        with state_offload_manger(  
            strategy=self.edit_strategy,  
            metrics=metrics,  
            metric_infix=f"{self.cluster_name}/generating_edited_image",  
            #is_offload_states=is_offload_states,  
            is_offload_states=self.is_offload_state_edit_model,     # 使用新的控制变量
        ):  
            self.logger.info(f"Generating edited images for global_step {global_step}...")
            edited_images = self._generate_t2i_images(data)  
        time_1 = time.time()
        self.logger.info("[ReasonT2IRewardWorker] Image generating took: {:.3f}s".format(time_1 - time_0))
        
        torch.cuda.empty_cache()

        
        # 阶段2: 使用 VLM Model 对编辑图片进行打分  
        with state_offload_manger(  
            strategy=self.vlm_strategy,  
            metrics=metrics,  
            metric_infix=f"{self.cluster_name}/vlm_scoring",  
            #is_offload_states=is_offload_states,  
            is_offload_states=self.is_offload_state_vlm,            # 使用新的控制变量 
        ):  
            self.logger.info(f"Scoring edited images for global_step {global_step}...")
            scores = self._score_t2i_images(edited_images, data)  
        time_2 = time.time()
        self.logger.info("[ReasonT2IRewardWorker] Image scoring took: {:.3f}s".format(time_2 - time_1))
        
        torch.cuda.empty_cache()
        
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )
        output.meta_info = {"metrics": metrics}
        self.logger.info(f"[ReasonT2IRewardWorker] Finished reward computation for global_step {global_step}. Batch rewards: {scores}")
        return output
            