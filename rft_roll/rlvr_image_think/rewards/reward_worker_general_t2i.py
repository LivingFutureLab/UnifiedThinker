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


# 参考 prism-bench
def get_message_templates():
    messages_1 = """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against its text prompt using a strict, two-step process. You will provide a one-sentence justification and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
Core Principle: The primary criterion is always Text-Image Alignment. The image must first be a faithful depiction of the literal content described in the prompt. The evaluation of the emotional aspect is a secondary, but important, step.
9-10 (Exceptional): Flawless. The image perfectly depicts all literal content from the prompt AND masterfully visualizes the specified emotion with depth and creativity.
7-8 (Good): The image depicts all literal content correctly, AND the emotional visualization is strong and accurate.
5-6 (Average): A competent attempt. The image depicts the literal content correctly, but the emotional visualization is weak, superficial, or relies heavily on clichés.
3-4 (Poor): Major failure in content alignment. Key subjects, objects, or settings from the prompt are missing or wrong. The emotional evaluation is largely irrelevant because the core content is incorrect.
0-2 (Failure): The image shows no significant resemblance to the literal content of the prompt.

Track-Specific Instructions: A Two-Step Evaluation
You must follow this sequence. Start at 10 and deduct points for each failure.
Step 1: Verify Content Alignment (Primary Criterion)
First, ignore the emotional component and check only the physical description. Does the image contain the correct subjects, objects, setting, and actions?
Content Mismatch (-6 to -8 points): This is the most severe failure. The image is missing a key subject, setting, or object described in the prompt. If the core content is wrong, the score cannot be high.
Attribute Error (-3 to -5 points): The content is generally right, but key attributes are wrong.
Step 2: Evaluate Emotional Visualization (Secondary Criterion)
Only after confirming the content alignment, evaluate the emotional layer.
Emotional Dissonance (-3 to -5 points): The image content is correct, but the mood is completely wrong. The lighting, colors, and composition fail to evoke the requested emotion.
Missing Nuance / Clichéd Symbolism (-2 to -4 points): The content is correct, but the emotion is handled superficially. The image uses an obvious cliché without any depth, or it captures a generic version of the emotion.
Literal Interpretation of Emotion (-2 to -4 points): The content is correct, but the emotion is interpreted in a clumsy, literal way.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_2 =  """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against its text prompt, focusing on object count and spatial relationships. You will provide a one-sentence justification and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. Every object, count, attribute, and spatial relationship is rendered with perfect accuracy and logical consistency.
7-8 (Good): The main objects and their primary relationships are correct. There might be a single, minor error in a secondary object's attribute or position.
5-6 (Average): A competent attempt. The image contains the correct primary objects, but there are significant errors in their count, spatial relationships, or interactions.
3-4 (Poor): Major errors in object count or the relationships between primary objects. The scene is fundamentally incorrect.
0-2 (Failure): The wrong objects are depicted, or the image is completely unrelated to the prompt.

Track-Specific Instructions: Object Layout and Relationships
Start at 10 and deduct points for each failure. Be systematic.
Incorrect Object Count (-3 to -5 points): The number of a key object is wrong.
Incorrect Spatial Relationship (-3 to -5 points): The relative position of key objects is wrong.
Incorrect Object Attributes (-2 to -4 points): A key object has the wrong color, size, or other specified attribute.
Incorrect Interactions (-2 to -4 points): A described interaction between objects or subjects is missing or wrong.
Minor Positional/Attribute Errors (-1 to -2 points): A secondary object is slightly misplaced or has a minor incorrect attribute.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_3 = """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt naming a specific entity. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The entity is rendered with photographic accuracy, and the surrounding scene perfectly matches all details in the prompt.
7-8 (Good): The entity is highly recognizable and accurate, and the overall scene is a good match for the prompt with only minor deviations.
5-6 (Average): A competent attempt. The entity is recognizable but has clear flaws, OR the entity is perfect but the surrounding scene described in the prompt is incorrect. An accurate entity in a wrong context is not a success.
3-4 (Poor): The entity is barely recognizable or is a generic substitute. The scene is also likely incorrect.
0-2 (Failure): The entity is wrong or absent, and the image is unrelated to the prompt.

Track-Specific Instructions: Specific Entity Generation
Start at 10 and deduct points for each failure. Prioritize overall alignment, then entity accuracy.
Incorrect Scene/Context (-4 to -6 points): The entity is correct, but the background, style, or action described in the prompt is completely wrong. This is a major failure.
Unrecognizable or Flawed Entity (-3 to -5 points): The entity is poorly rendered, has significant anatomical or structural errors, or looks like a generic version.
Missing Scene Details (-2 to -4 points): The scene is generally correct, but key descriptive elements are missing.
Minor Entity Inaccuracies (-1 to -3 points): The entity is recognizable but has small, specific inaccuracies.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_4 =  """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt describing an imaginative object. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. All described features are seamlessly and creatively integrated into a coherent, believable whole. The object feels truly unique and masterfully executed.
7-8 (Good): The object is well-designed and incorporates almost all key features from the prompt with good coherence.
5-6 (Average): A competent attempt. The object includes the main features described, but they appear "stitched together" or incoherent. Key details are missing or misinterpreted. The result is a recognizable but flawed collage of ideas.
3-4 (Poor): The object is a confusing mess, missing most of the core features described in the prompt.
0-2 (Failure): The object is completely wrong or the image is unrelated to the prompt.

Track-Specific Instructions: Imaginative Object Generation
Start at 10 and deduct points for each failure. Focus on coherence.
Missing Core Features (-4 to -6 points): Fails to include a defining feature of the object.
Lack of Coherence (-3 to -5 points): The described parts are present but look like a poorly assembled collage rather than a single, integrated object.
Misinterpreted Attributes (-2 to -4 points): A key material or quality is rendered incorrectly.
Incorrect Context (-1 to -3 points): The object is rendered well, but the surrounding environment described in the prompt is wrong.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_5 =  """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt requesting a specific style. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The image perfectly captures the content and executes the requested style with deep, nuanced understanding of its aesthetics, techniques, and historical context.
7-8 (Good): The content is correct, and the style is clearly recognizable and well-executed, with only minor deviations from the style's core principles.
5-6 (Average): A competent but superficial attempt. The content is correct, but the style is applied like a simple filter. It captures the most obvious stylistic clichés but misses the nuance of the art form.
3-4 (Poor): The content is correct but the style is wrong, OR the style is vaguely correct but the content is wrong.
0-2 (Failure): Both content and style are wrong.

Track-Specific Instructions: Specific Style Application
Start at 10 and deduct points for each failure. Penalize superficiality.
Incorrect Content (-5 to -7 points): The image shows the wrong subject matter, even if the style is correct. This is a major failure.
Superficial Style Application (-4 to -6 points): The image uses only the most obvious clichés of a style without understanding its underlying principles.
Missing Stylistic Elements (-2 to -4 points): The image misses key technical identifiers of the style.
Inconsistent Style (-1 to -3 points): Parts of the image are in the correct style while other parts are not.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_6 =  """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image that should contain rendered text. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The text is perfectly spelled, legible, and seamlessly integrated into the scene with correct perspective, lighting, and texture.
7-8 (Good): The text is perfectly spelled and legible, with only very minor issues in its integration.
5-6 (Average): A competent attempt. The text is spelled correctly but is poorly integrated into the scene. It may look flat, have unnatural lighting, or be placed awkwardly.
3-4 (Poor): The text contains significant spelling errors or is partially illegible, even if the placement is roughly correct.
0-2 (Failure): The text is nonsensical, completely wrong, or absent.

Track-Specific Instructions: In-Image Text Generation
Start at 10 and deduct points for each failure. Text accuracy is paramount.
Spelling or Wording Errors (-6 to -8 points): Any deviation from the requested text string. This is the most severe failure.
Poor Integration (-3 to -5 points): The text looks pasted on, with incorrect perspective, lighting, or shadows for the scene.
Illegibility (-3 to -5 points): The characters are garbled, distorted, or difficult to read.
Incorrect Placement/Font (-2 to -4 points): The text is on the wrong object or in the wrong location, or the requested font style is ignored.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_7 = """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a long, detailed text prompt. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The image comprehensively and coherently visualizes virtually every detail from the prompt, from major elements to minor attributes.
7-8 (Good): The image captures all major elements and a clear majority of the secondary details and attributes. The omissions are minor.
5-6 (Average): A competent attempt. The image correctly depicts the main subject and setting but omits a significant number of secondary details and attributes. The core is there, but the richness is lost.
3-4 (Poor): The image captures only one of the major elements and misses almost all descriptive details.
0-2 (Failure): The image fails to capture any of the major elements described in the prompt.

Track-Specific Instructions: Long Text Comprehension
Start at 10 and deduct points for each failure. Be a detail-oriented critic.
First, identify the Major Elements (primary subject, setting, main action).
Second, list all Secondary Details (other objects, characters, specific attributes).
Deduct points for each omission or error.
Missing a Major Element (-5 to -7 points): Fails to include the primary subject, setting, or action.
Missing a Majority of Secondary Details (-3 to -5 points): The image feels generic because it ignored most of the specific descriptors that gave the prompt its character.
Incorrectly Rendered Detail (-2 to -4 points): A detail is included but rendered incorrectly.
Each Minor Omission (-1 point): For every small, specific detail that is missing, deduct a point.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    messages_8 = """ 
You are a hyper-critical quality assurance inspector for a text-to-image generation benchmark. Your task is to evaluate images with forensic, microscopic scrutiny. Your primary directive is to penalize any deviation from physical, anatomical, and logical coherence, unless such deviations are explicitly requested by the text prompt. Assume all subjects and environments must be perfectly sound and plausible by default.

Scoring System: You will start with a perfect score of 10 and deduct points for any flaws you identify. A single significant flaw should prevent a high score.

Flaw Categories (Deduct points for each instance):
Critical Failures (-7 to -9 points):
Any violation of the fundamental anatomical or structural integrity of the main subjects. This includes inconsistencies in form, function, or natural appearance.
A breakdown in logical or physical plausibility within the scene, when not specified by the prompt.
Prominent, distracting digital artifacts, watermarks, or signatures that ruin immersion.
The central subject is rendered as grotesque or nonsensical, when not specified by the prompt.
Significant Flaws (-4 to -6 points):
Noticeable warping, distortion, or a lack of convincing texture on key objects or surfaces.
Unnatural blending, texture repetition, or other clear indicators of AI synthesis that break realism.
Lack of sharpness or resolution in the primary subject, making crucial details indistinct.
Incoherent or illogical features on secondary elements.
Minor Imperfections (-1 to -3 points):
Slight compositional awkwardness or minor issues with lighting and shadow that don't break realism.
Minimal blurriness or noise in secondary, non-focal areas of the image.
Faint, non-distracting artifacts that are only visible upon close inspection.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a “score“:
{{
"justification": …,
"score": …,
}}

text prompt: {text_prompt}
"""

    return {
        "alignment":{
            "affection": messages_1,
            "composition": messages_2,
            "entity": messages_3,
            "imagination": messages_4,
            "style": messages_5,
            "text_rendering": messages_6,
            "long_text": messages_7,
        },
        "aesthetic": messages_8,
    }

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

def clean_and_parse_json(json_str: str) -> Dict[str, Any]:
    """
    Cleans and parses a JSON string, with fallback to demjson3.
    """
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    json_str = re.sub(r",\s*(?=[}\]])", "", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Standard JSON parsing failed, falling back to demjson3.")
        try:
            return demjson3.decode(json_str)
        except demjson3.JSONDecodeError as e:
            print(f"Failed to parse JSON with demjson3: {e}")
            return {}
       
        
class GeneralT2IRewardWorker(Worker):
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
                "Configuration error: Expected 'clusters' list with at least 2 entries in the YAML file for GeneralT2IRewardWorkerV2."
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
            breakpoint()
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
                self.logger.info("[GeneralT2IRewardWorker] FUll prompt_cot: {}".format(repr(prompt_cot)))
                prompt_cot = match.group(1).strip()
                self.logger.info("[GeneralT2IRewardWorker] Answer of prompt_cot: {}".format(repr(prompt_cot))) 
            
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
                self.logger.info(f"[GeneralT2IRewardWorker] [qwen-image-edit-think] - generate image with prompt: {prompt};  prompt_cot: {prompt_cot}")
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
                self.logger.info(f"[GeneralT2IRewardWorker] [qwen-image-edit] (origin prompt: {prompt})- generate image with prompt_cot: {prompt_cot}")
                
            output = edit_model(**inputs, height=height, width=width)
            output_image = output.images[0]
            edited_images.append(output_image)
        return edited_images
    
    @torch.no_grad()
    def _score_t2i_images(self, edited_images, data: DataProto):
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            breakpoint()
            self.logger.info("进入 Debug 模式: _score_t2i_images")
            
        vlm_model = self.vlm_strategy.unwrap_model() # 或者 self.vlm_strategy.model
        messages_pool = get_message_templates()
        
        scores = []
        for i, (image, prompt, domain, t2i_eval_category) in enumerate(zip(
            edited_images, data.non_tensor_batch["edit_prompt"], data.non_tensor_batch["domain"], data.non_tensor_batch["t2i_eval_category"]
        )):
            try:
                assert "t2i" in domain, "wrong data domain, this reward worker is used for t2i task."
                assert isinstance(image, Image.Image), "post_edit_image is not Image"
                
                results = []
                for eval_type in ["alignment", "aesthetic"]:
                    if eval_type == "alignment":
                        messages_template = messages_pool.get("alignment", {}).get(t2i_eval_category)
                    else: # aesthetic
                        messages_template = messages_pool.get("aesthetic")

                    image = image.convert("RGB")
                    final_prompt = messages_template.format(text_prompt=prompt)
                    
                    messages = [{
                        "role": "user",
                        "content": [{"type": "image", "image": image},
                                    {"type": "text", "text": final_prompt}]
                    }]
                    output_text = qwen_vl_generate(vlm_model, messages)
                    
                    result_data = clean_and_parse_json(output_text)
                    results.append(result_data['score'])
                
                score_align, score_aes = results
                final_score = (score_align + score_aes) / 2.0
            except Exception as e:
                print("error of {} during _scroe_t2i_images".format(str(e)))
                final_score = 0.0
            scores.append(final_score)
                
        return scores
        
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)  
    def compute_rewards(self, data: DataProto):  
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            breakpoint()
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
        self.logger.info("[GeneralT2IRewardWorker] Image generating took: {:.3f}s".format(time_1 - time_0))

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
        self.logger.info("[GeneralT2IRewardWorker] Image scoring took: {:.3f}s".format(time_2 - time_1))
        
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
        self.logger.info(f"[GeneralT2IRewardWorker] Finished reward computation for global_step {global_step}. Batch rewards: {scores}")
        return output
            