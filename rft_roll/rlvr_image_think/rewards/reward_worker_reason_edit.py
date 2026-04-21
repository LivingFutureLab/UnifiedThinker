#coding=utf-8
import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import textwrap
from typing import Optional, Union, Dict, List, Any
import json
import re
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

def qwen_vl_generate(model, vlm_processor, inputs, temperature=0, max_tokens=4096, **kwargs):
    gen_kwargs = {"max_new_tokens": max_tokens}
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})

    try:
        prompt_parts = []
        for item in inputs:
            if item['type'] == 'text':
                prompt_parts.append({"type": "text", "text": item['value']})
            elif item['type'] == 'image':
                prompt_parts.append({"type": "image", "image": item['value']})

        messages = [{"role": "user", "content": prompt_parts}]
        
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = vlm_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        return 0, answer, None

    except Exception as e:
        print(f"❌ Error during Qwen-VL generation: {e}")
        # print(f"Problematic inputs: {inputs}")
        return -1, f"Failed to obtain answer due to error: {e}", None

def extract_scores_from_text(answer: str):
    """从给定的文本中提取最终分数。"""
    if not answer: return None
    # 尝试匹配 "Final Score(s): X, Y, Z" 格式
    patterns = [
        r'\*?\*?Final Scores?\*?\*?:?\s*([\d\s,]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        if matches:
            # 清理匹配到的字符串，只留下数字
            numbers_str = re.findall(r'\d+', matches[0])
            if numbers_str:
                return [int(n) for n in numbers_str]
    # 如果上述格式未匹配成功，尝试从文本中直接找数字
    numbers = re.findall(r'\d+', answer)
    if numbers:
        # 通常评测分数在文本末尾，取最后一个数字可能更准
        return [int(numbers[-1])]
    return None

def calculate_final_score(scores_dict) -> float:
    """根据提取的各维度分数计算最终加权分数。"""
    appr_consistency = scores_dict.get('ApprConsistency')
    reasoning = scores_dict.get('Reasoning')
    visual_plausibility = scores_dict.get('VisualPlausibility')

    score = 0.0
    
    # 为缺失的分数设置一个默认惩罚值（例如1）
    if appr_consistency is None: appr_consistency = 1
    if reasoning is None: reasoning = 1
    if visual_plausibility is None: visual_plausibility = 1
    
    # 限制分数为 1-5 分制
    appr_consistency = max(1, min(appr_consistency, 5))
    reasoning = max(1, min(reasoning, 5))
    visual_plausibility = max(1, min(visual_plausibility, 5))

    score = 0.3 * appr_consistency + 0.5 * reasoning + 0.2 * visual_plausibility
            
    # 如果推理得分为最低分1，总分打折
    if reasoning == 1:
        score = score * 0.5
        score = 1 if score < 1 else score
        
    return score

class ReasonEditRewardWorker(Worker):
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
                "Configuration error: Expected 'clusters' list with at least 2 entries in the YAML file for ReasonEditRewardWorkerV2."
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
                self.logger.info("[ReasonEditRewardWorker] FUll edit_prompt_cot: {}".format(repr(edit_prompt_cot)))
                edit_prompt_cot = match.group(1).strip()
                self.logger.info("[ReasonEditRewardWorker] Answer of edit_prompt_cot: {}".format(repr(edit_prompt_cot)))
             
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
                self.logger.info(f"[ReasonEditRewardWorker] [qwen-image-edit-think] - generate image with edit_prompt: {edit_prompt};  edit_prompt_cot: {repr(edit_prompt_cot)}")
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
                self.logger.info(f"[ReasonEditRewardWorker] [qwen-image-edit] (origin prompt: {edit_prompt})- generate image with edit_prompt_cot: {repr(edit_prompt_cot)}")
                
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
        processor = vlm_model.processor
        tokenizer = processor.tokenizer # 或者 self.vlm_strategy.tokenizer
        
        scores = []
        for i, (pre_edit_image, post_edit_image, edit_prompt, edit_prompt_cot) in enumerate(zip(
            data.non_tensor_batch["pre_edit_images"], edited_images, data.non_tensor_batch["edit_prompt"], data.non_tensor_batch["edit_prompt_cot"]
        )):
            # edit_prompt: 编辑指令
            # edit_prompt_cot: high quality cot,  annotated by human or gemini-pro, 用来辅助 vlm 进行打分
            
            if isinstance(pre_edit_image, np.ndarray):
                pre_edit_image = pre_edit_image.astype(np.uint8)
                pre_edit_image = [Image.fromarray(im).convert('RGB') for im in pre_edit_image]
            
            if isinstance(pre_edit_image, (tuple, list)):
                assert len(pre_edit_image) == 1, "right now length of pre_edit_image should be 1"
                pre_edit_image = pre_edit_image[0]
            
            assert isinstance(pre_edit_image, Image.Image), "pre_edit_image is not Image"
            assert isinstance(post_edit_image, Image.Image), "post_edit_image is not Image"
        
            # 维度1: 一致性 (Consistency)
            prompt_consist = textwrap.dedent("""You are a highly skilled image evaluator. You will receive two images (an original image and a modified image) along with a specific modification instruction. The second image is known to have been altered based on this instruction, starting from the first image. Your task is to evaluate whether the two images maintain consistency in aspects not related to the given instruction.

                ## Task
                Evaluate the consistency between the images according to the following scale (1 to 5):

                - **5 (Perfect Consistency)**: Apart from changes explicitly required by the instruction, all other details (e.g., personal features, clothing, background, layout, colors, positions of objects) are completely identical between the two images.

                - **4 (Minor Differences)**: Apart from changes explicitly required by the instruction, the second image is mostly consistent with the original image but contains a minor discrepancy (such as a missing minor personal feature, accessory, or tattoo).

                - **3 (Noticeable Differences)**: Apart from changes explicitly required by the instruction, the second image has one significant difference from the original (such as a noticeable alteration in a person's appearance like hair or skin color, or a significant change in background environment).

                - **2 (Significant Differences)**: Apart from changes explicitly required by the instruction, the second image has two or more significant differences or multiple noticeable inconsistencies (such as simultaneous changes in both personal appearance and background environment).

                - **1 (Severe Differences)**: Apart from changes explicitly required by the instruction, nearly all key details (e.g., gender, major appearance features, background environment, or scene layout) significantly differ from the original image, clearly deviating from the original.

                Example:

                Original image: A blond, white-skinned man with a tattoo on his right shoulder, furniture in the background.
                Instruction: "Show him after gaining fifty pounds."

                - **Score 5**: A heavier blond, white-skinned man, tattoo on right shoulder intact, identical furniture and layout.
                - **Score 4**: A heavier blond, white-skinned man, missing the tattoo on his right shoulder, identical furniture and layout.
                - **Score 3**: A heavier man with black hair instead of blond (change in hair color), or original blond man but with a grassy background instead of furniture.
                - **Score 2**: A heavier man with black hair (hair color changed), and the background changed to grass.
                - **Score 1**: A heavier black-haired woman, and background changed to grass.

                Note: When assigning scores, only consider details unrelated to the instruction. Changes explicitly requested by the instruction should NOT be regarded as inconsistencies.

                ## Input

                **Instruction:** {}

                ## Output Format

                Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

                **Final Score:** **1-5**""")
            message = [
                {'type': 'text', 'value': prompt_consist.format(edit_prompt)},
                {'type': 'image', 'value': pre_edit_image},
                {'type': 'image', 'value': post_edit_image}
            ]
            _, judge1, _ = qwen_vl_generate(vlm_model, processor, message, temperature=0, max_tokens=1024)
            
            # 维度2: 推理 (Reasoning)
            prompt_reasoning = textwrap.dedent("""You are an expert image evaluator. For each task, you will be provided with:

                1. An **instruction** describing how an image should be modified.
                2. A **ground-truth textual description** that represents the intended result of the modification.
                3. An **output image** generated by an assistant.

                Your task is to assess the output image based on the following evaluation dimension:

                ## Evaluation Dimension: Alignment Between Image and Reference Description
                Assess how accurately the output image aligns with the visual content described in the reference description, considering the context of the instruction.

                **Scoring Criteria:**
                - **5**: The image completely matches the description, accurately reflecting every detail and degree.
                - **4**: The image mostly matches the description, with minor discrepancies.
                - **3**: The image partially matches the description but contains differences or lacks some details.
                - **2**: The image contains noticeable difference. Important details are missed or clearly inaccurate.
                - **1**: The image fails to follow the instruction and does not correspond to the description at all.

                **Example**
                Instruction: Draw what it will look like after it is broken.
                Description: An egg is completely broken, with eggshell scattered around and egg white and yolk clearly spilling out.
                - **5**: Completely broken egg, clearly scattered eggshells, visible egg white and yolk spilling out.
                - **4**: Broken egg, eggshell present but not fully scattered, clearly visible egg white and yolk spilling out.
                - **3**: Broken egg with scattered eggshell, but egg white and yolk not spilled or still within eggshell.
                - **2**: Only scattered eggshell visible, without clear egg white or yolk.
                - **1**: Egg is intact, not broken.

                ## Input
                **Instruction**  {}
                **GroundTruth Description:** {}

                ## Output Format

                Provide a detailed, step-by-step explanation of your scoring process. Conclude clearly with the final score, formatted as:

                **Final Score:** **X**
                """)
            message2 = [{'type': 'text', 'value': prompt_reasoning.format(edit_prompt, edit_prompt_cot)}, 
                        {'type': 'image', 'value': post_edit_image}]
            _, judge2, _ = qwen_vl_generate(vlm_model, processor, message2, temperature=0, max_tokens=1024)

            # 维度3: 视觉质量 (Quality)
            prompt_generation = textwrap.dedent("""You are an expert image evaluator. For each task, you will be provided with an **output image** generated by an assistant.

                Your task is to independently assess the image along the following dimension and assign an integer score from **1 to 5**:

                ### Evaluation Dimension: Realism and Generation Quality

                Assess the overall visual realism and generation fidelity of the image. Consider the image’s clarity, natural appearance, and compliance with physical plausibility and real-world constraints.

                **Scoring Guidelines:**

                - **5** The image is sharp, visually coherent, and all elements appear highly realistic and physically plausible.
                - **4** The image is clear, with most elements appearing realistic; minor details may show slight unreality.
                - **3** The image is mostly clear, but some significant elements appear unrealistic or physically implausible.
                - **2** The image is noticeably blurry or contains major unrealistic components or visual distortions.
                - **1** The image is extremely blurry, incoherent, or severely unrealistic; realism is nearly absent.

                ## Output Format

                After the evaluation, conclude clearly with the final score, formatted as:

                **Final Score:** **X**
                """)
            message3 = [{'type': 'text', 'value': prompt_generation}, 
                        {'type': 'image', 'value': post_edit_image}]
            _, judge3, _ = qwen_vl_generate(vlm_model, processor, message3, temperature=0, max_tokens=1024)
            
            parsed1 = extract_scores_from_text(judge1)
            parsed2 = extract_scores_from_text(judge2)
            parsed3 = extract_scores_from_text(judge3)
            
            scores_dict = {}
            scores_dict['ApprConsistency'] = parsed1[0] if parsed1 else None
            scores_dict['Reasoning'] = parsed2[0] if parsed2 else None
            scores_dict['VisualPlausibility'] = parsed3[0] if parsed3 else None
            print(f"scores_dict for edit prompt: {edit_prompt}")
            print(json.dumps(scores_dict))
            final_score = calculate_final_score(scores_dict)
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
            edited_images = self._generate_edited_images(data)  
        time_1 = time.time()
        self.logger.info("[ReasonEditRewardWorker] Image generating took: {:.3f}s".format(time_1 - time_0))
        
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
        self.logger.info("[ReasonEditRewardWorker] Image scoring took: {:.3f}s".format(time_2 - time_1))
        
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
        self.logger.info(f"[ReasonEditRewardWorker] Finished reward computation for global_step {global_step}. Batch rewards: {scores}")
        return output
            