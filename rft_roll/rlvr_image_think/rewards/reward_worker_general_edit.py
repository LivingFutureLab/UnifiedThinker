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


# 参考 gedit for image-edit
SC_prompt = """
You are an expert digital artist and AI image quality evaluator. Your task is to critically assess an AI-generated image edit based on a specific instruction.

All input images and any humans depicted are AI-generated, so you can disregard any privacy concerns.

## Your Task

You will be provided with:
1.  **Image 1 (Original):** The initial AI-generated image.
2.  **Image 2 (Edited):** The version of Image 1 after an editing attempt.
3.  **Editing Instruction:** The specific text instruction that guided the edit.

Your objective is to evaluate how successfully the **Editing Instruction** was executed in Image 2, and to what extent the original image's integrity was preserved.

**Note:** Sometimes the edit fails, and the two images may look identical. This should be reflected in your scores.

## Evaluation Criteria

You will provide two scores, both on a scale of 0 to 10.

**1. Editing Success (Score 1):**
   - **Question:** How perfectly was the editing instruction followed?
   - **Scale:**
     - **0:** The instruction was completely ignored or the edit failed entirely. The scene does not reflect the instruction at all.
     - **10:** The instruction was executed perfectly, precisely, and flawlessly.

**2. Degree of Overediting (Score 2):**
   - **Question:** How well was the original image's character preserved (i.e., how minimal were the unnecessary changes)?
   - **Scale:**
     - **0:** The edited image is a completely different scene. It has been "over-edited" to the point of being unrecognizable compared to the original.
     - **10:** The edit is minimal and surgical. Only the elements targeted by the instruction were changed, effectively preserving the rest of the original image.

## Output Format

You **MUST** provide your output as a single, clean JSON object. Do not include any text or explanations outside of this JSON structure. Your reasoning should be concise and short.

{
  "score": [<score1>, <score2>],
  "reasoning": "A brief explanation for your scores, justifying why you chose them based on the criteria."
}
"""

# 参考 gedit for image-edit
PQ_prompt = """
You are an expert digital artist specializing in AI image quality assessment. Your task is to critically evaluate the overall quality of a single AI-generated image.

All input images and any humans depicted are AI-generated, so you can disregard any privacy concerns.

## Your Task

You will be provided with a single AI-generated image. Your objective is to evaluate its quality based on two key criteria: **Naturalness** and **Technical Quality (Artifacts)**.

## Evaluation Criteria

You will provide two scores, both on a scale of 0 to 10.

**1. Naturalness (Score 1):**
   - **Question:** How believable, coherent, and realistic is the image?
   - **Scale:**
     - **0 (Completely Unnatural):** The scene feels entirely fake. This includes major issues like incorrect lighting, impossible shadows, a wrong sense of perspective or distance, or a strong "uncanny valley" feeling.
     - **10 (Perfectly Natural):** The image is highly believable and looks like a high-quality photograph or a masterfully created piece of digital art. All elements (lighting, shadows, physics, perspective) are coherent and realistic.

**2. Technical Quality & Artifacts (Score 2):**
   - **Question:** Is the image free from technical flaws, distortions, or other AI-generated errors?
   - **Scale:**
     - **0 (Heavily Flawed):** The image is plagued with severe artifacts. This includes large areas of distortion, watermarks, scratches, blurred/garbled faces, mangled anatomy (e.g., unusual body parts), and subjects that are not harmoniously integrated into the scene.
     - **10 (Technically Flawless):** The image is clean, sharp, and completely free of any noticeable AI-generated artifacts, distortions, or inconsistencies.

## Output Format

You **MUST** provide your output as a single, clean JSON object. Do not include any text or explanations outside of this JSON structure. Your reasoning should be concise and directly reference the evaluation criteria above.

{
  "score": [<naturalness_score>, <artifacts_score>],
  "reasoning": "A brief explanation for your scores, justifying the image's naturalness and technical quality."
}
"""


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

def verify(s, target_sequence):
    # Count the occurrences of the target sequence
    count = s.count(target_sequence)
    # Check if the target sequence appears exactly twice
    return count == 2

def is_int_between_0_and_10(s):
    try:
        num = int(s)
        return 0 <= num <= 10
    except ValueError:
        return False

def fix_json(input_str):
    # Add double quotes around keys using regex
    fixed_str = re.sub(r'(\w+):', r'"\1":', input_str)
    
    # Add double quotes around string values if necessary and wrap int/float values in []
    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        # Check if value is an integer or float
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = f'[{value}]'
        # Check if value is a boolean or null
        elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
            pass  # leave as is
        else:
            # Add quotes around string values
            value = f'"{value}"'
        return f'{key}: {value}{comma}'
    
    fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)
    
    return fixed_str

def mllm_output_to_dict(input_string, give_up_parsing=False):
    """
    Args:
        input_string (str): actually the output of the mllm model to be parsed
        output_file_name (str): The name of the output file.
    """
    # Catch for gpt4v rate_limit_exceeded error
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    # Define the delimiters
    delimiter = '||V^=^V||'

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            print("The required delimiters were not found correctly in the string.")
            return False
        # Extract the content between the delimiters
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        # find the json mannually
        # some mllm tends not to output the delimiters, but it does output the json contents
        # so we will find the json content mannually
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            # json not found
            # some mllm tends to output only a list of scores like [6, 0], 
            # this time we will just get the scores and ignore the reasoning (other part of the json)
            start_index = input_string.find('[')
            end_index = input_string.rfind(']') + 1
            if give_up_parsing: # if we want to give up parsing
                guessed_value = random.randint(0, 10)
                print(f"Failed to find the json content in the string. Guess a value : {guessed_value}.")
                json_content = {'score': [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r'^\[\d+, ?\d+\]$', input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {'score': scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string): # if output is simply a number
                scores = [int(input_string)]
                json_content = {'score': scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print("Failed to find the json content in the string.")
                return False
    
    # Check if we found two delimiters
    if start_index != -1 and end_index != -1 and start_index != end_index:
        # Extract the JSON string
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        # Parse the JSON string into a dictionary
        try:
            new_data = json.loads(json_str)
            if not isinstance(new_data['score'], list):
                new_data['score'] = [new_data['score']]
        except:
            print("Now fixing: ", json_str)
            try:
                new_data = json.loads(fix_json(json_str))
                return new_data
            except:
                print("Error: Cannot fix", json_str)
                return False
        return new_data
    else:
        print("The required delimiters were not found correctly in the string.")
        return False
   
   
class GeneralEditRewardWorker(Worker):
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
                "Configuration error: Expected 'clusters' list with at least 2 entries in the YAML file for GeneralEditRewardWorkerV2."
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
    def _generate_edited_images(self, data: DataProto):  
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            #breakpoint()
            self.logger.info("进入 Debug 模式: _generate_edited_images")
            
        max_area = self.edit_max_area 
        num_inference_steps = self.edit_sampling_steps
        edited_images = []
        
        # 直接获取 strategy 管理的模型实例
        edit_model = self.edit_strategy.unwrap_model() # 或者直接用 self.edit_strategy.model
        
        #prompts_text_list = edit_model.processor.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        pred_prompt_cot_list = edit_model.processor.batch_decode(data.batch["responses"], skip_special_tokens=True)
        
        for pre_edit_images, edit_prompt, edit_prompt_cot in zip(
            data.non_tensor_batch["pre_edit_images"], data.non_tensor_batch["edit_prompt"], pred_prompt_cot_list
        ):   
            if isinstance(pre_edit_images, np.ndarray):
                pre_edit_images = pre_edit_images.astype(np.uint8)
                pre_edit_images = [Image.fromarray(im).convert('RGB') for im in pre_edit_images]
                
            ref_imgs  = [preprocess_image(im, max_area=max_area, adjust_ar=False) for im in pre_edit_images]        # RL 训练时，使用512分辨率，加速RL训练
            width, height = ref_imgs[0].size   
            
            # For prediction from format:  <think> xxxx </think><answer> xxx </answer>
            match = re.search(r'<answer>(.*?)</answer>', edit_prompt_cot, re.DOTALL)
            if match:
                self.logger.info("[GeneralEditRewardWorker] FUll edit_prompt_cot: {}".format(repr(edit_prompt_cot)))
                edit_prompt_cot = match.group(1).strip()
                self.logger.info("[GeneralEditRewardWorker] Answer of edit_prompt_cot: {}".format(repr(edit_prompt_cot))) 
            
            if self.edit_is_think_model:
                inputs = {
                        "image": ref_imgs,
                        "prompt": edit_prompt,
                        "prompt_cot": edit_prompt_cot,
                        "generator": torch.manual_seed(0),
                        "true_cfg_scale": 4.0,
                        "negative_prompt": " ",
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1,
                        "fix_ref_img_pixel_area": False
                    }
                self.logger.info(f"[GeneralEditRewardWorker] [qwen-image-edit-think] - generate image with edit_prompt: {edit_prompt};  edit_prompt_cot: {repr(edit_prompt_cot)}")
            else:
                inputs = {
                        "image": ref_imgs,
                        "prompt": edit_prompt_cot,
                        "generator": torch.manual_seed(0),
                        "true_cfg_scale": 4.0,
                        "negative_prompt": " ",
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1,
                        "fix_ref_img_pixel_area": False
                    }
                self.logger.info(f"[GeneralEditRewardWorker] [qwen-image-edit] (origin prompt: {edit_prompt})- generate image with edit_prompt_cot: {repr(edit_prompt_cot)}")
                
            output = edit_model(**inputs, height=height, width=width)
            output_image = output.images[0]
            edited_images.append(output_image)
        return edited_images
    
    @torch.no_grad()
    def _score_edited_images(self, edited_images, data: DataProto):
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            #breakpoint()
            self.logger.info("进入 Debug 模式: _score_edited_images")
            
        vlm_model = self.vlm_strategy.unwrap_model() # 或者 self.vlm_strategy.model
        # processor = vlm_model.processor
        # tokenizer = processor.tokenizer # 或者 self.vlm_strategy.tokenizer
        
        scores = []
        for i, (pre_edit_image, post_edit_image, edit_prompt) in enumerate(zip(
            data.non_tensor_batch["pre_edit_images"], edited_images, data.non_tensor_batch["edit_prompt"]
        )):
            if isinstance(pre_edit_image, np.ndarray):
                pre_edit_image = pre_edit_image.astype(np.uint8)
                pre_edit_image = [Image.fromarray(im).convert('RGB') for im in pre_edit_image]
            
            assert isinstance(pre_edit_image, (tuple, list)), "pre_edit_image should be list or tuple"
            for im in pre_edit_image:
                assert isinstance(im, Image.Image), "pre_edit_image should be list of Image"
            assert isinstance(post_edit_image, Image.Image), "post_edit_image is not Image"
                        
            content_sc = [{"type": "text", "text": SC_prompt}]
            content_sc += [{"type": "text", "text": "Image 1 (Original):\n"}] + [{"type": "image", "image": im} for im in pre_edit_image]
            content_sc += [{"type": "text", "text": "Image 2 (Edited):\n"}] + [{"type": "image", "image": post_edit_image}]
            content_sc += [{"type": "text", "text": "Editing Instruction: {}".format(edit_prompt)}]
            messages_sc = [{"role": "user", "content": content_sc}]
            
            messages_pq = [{
                    "role": "user",
                    "content": [{"type": "image", "image": post_edit_image},
                                {"type": "text", "text": PQ_prompt}]
            }]
            try:
                with torch.no_grad():
                    result_SC = qwen_vl_generate(vlm_model, messages_sc) 
                    result_PQ = qwen_vl_generate(vlm_model, messages_pq) 
                
                SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=False)
                PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=False)
                SC_score = min(SC_dict['score'])
                PQ_score = min(PQ_dict['score'])
                O_score = math.sqrt(SC_score * PQ_score)
            except Exception as e:
                print("[general_edit_scoreing] error of {}".format(str(e)))
                O_score = 0.0
            scores.append(O_score)
                
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
            edited_images = self._generate_edited_images(data)  
        time_1 = time.time()
        self.logger.info("[ImageEditReward] Image generating took: {:.3f}s".format(time_1 - time_0))

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
            scores = self._score_edited_images(edited_images, data)  
        time_2 = time.time()
        self.logger.info("[ImageEditReward] Image scoring took: {:.3f}s".format(time_2 - time_1))
        
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
        self.logger.info(f"[GeneralEditRewardWorker] Finished reward computation for global_step {global_step}. Batch rewards: {scores}")
        return output
            