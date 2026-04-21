import os
import gc
from termcolor import colored
import math
import torch
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
# from safetensors.torch import load_model as load_model_safetensors
from safetensors.torch import load_file
from transformers import AutoProcessor
from safetensors import safe_open
import json 
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler

from src.model.utils import download_model_weight
from src.model.warp_model import WarpModel, WarpModel2
from src.utils.env_utils import (
    import_class,
    in_notebook
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration

from src.pipe.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline


def load_model(loader_func, loader_params):
    func = import_class(loader_func)
    model_dict = func(**loader_params)

    return model_dict

def prepare_model(
    model_dict,
    pre_func=None,
    pre_params={},
    warp_func=None,
    warp_params={},
    device="cuda",
    weight_dtype=torch.bfloat16,
):
    """Prepare model with optional pre-processing and warping functions.

    Args:
        model_dict: Dictionary containing model components
        pre_func: Optional pre-processing function
        pre_params: Parameters for pre-processing
        warp_func: Optional warping function
        warp_params: Parameters for warping
        device: Device to place model on
        weight_dtype: Data type for model weights
    """
    if pre_func is not None:
        func = import_class(pre_func)
        model_dict = func(
            model_dict, device=device, weight_dtype=weight_dtype, **pre_params
        )
    if warp_func is not None:
        func = import_class(warp_func)
        model_dict["train_model"] = func(model_dict, **warp_params)
    return model_dict


############################################ qwen-image-edit start #########################################
def load_model_weights(ckpt_path):
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict_transformer = {k.replace("transformer.", ""): v for k, v in state_dict.items() if k.startswith("transformer.")}
        state_dict_textencoder = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}
        assert len(state_dict) == len(state_dict_transformer) + len(state_dict_textencoder)
    elif os.path.isdir(ckpt_path):
        # 切片保存
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
    else:
        raise ValueError()
    return state_dict_transformer, state_dict_textencoder

def load_qwen_image_edit(model_path, qwenvl_path, edit_with_think=False):    
    vlm_processor = AutoProcessor.from_pretrained(qwenvl_path,
                                                min_pixels=64 * 28 * 28,
                                                max_pixels=1280 * 28 * 28)  # 用于 validation  
    if edit_with_think:
        pipe = QwenImageEditThinkPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        print(colored("Initialize QwenImageEditThinkPipeline", "green", attrs=["bold"]))
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        
    model_dict = {
        "transformer": pipe.transformer,
        "scheduler": pipe.scheduler,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "pipe": pipe,
        "vlm_processor": vlm_processor
    }
    return model_dict

def prepare_qwen_image_edit(
    model_dict,
    device="cuda",
    weight_dtype=torch.bfloat16,
    gradient_checkpointing_text_encoder=False,
    gradient_checkpointing_transformer=False,
    trainable_module=["transformer"],
    text_encoder_lora=False,
    transformer_lora=False,
    lora_r: int = 32,
    lora_alpha: int = 32,
    resume_ckpt_path: str = ""
):
    # vae
    model_dict["pipe"].vae.requires_grad_(False)
    
    # text_encoder
    model_dict["pipe"].text_encoder.requires_grad_(False)
    if "text_encoder" in trainable_module:
        if text_encoder_lora:
            text_encoder_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.01,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model_dict["pipe"].text_encoder.language_model.add_adapter(text_encoder_lora_config)
        else:
            model_dict["pipe"].text_encoder.requires_grad_(True)            # llm full trainable
            model_dict["pipe"].text_encoder.visual.requires_grad_(False)    # only freeze vit
    
    # dit
    model_dict["pipe"].transformer.requires_grad_(False)
    if "transformer" in trainable_module:
        if transformer_lora:
            transformer_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.01,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            model_dict["pipe"].transformer.add_adapter(transformer_lora_config)
        else:
            model_dict["pipe"].transformer.requires_grad_(True)             # dit full trainable

    if "transformer" in trainable_module and gradient_checkpointing_transformer:
        print("gradient_checkpointing for transformer enabled.")
        model_dict["transformer"].enable_gradient_checkpointing()
        
    if "text_encoder" in trainable_module and gradient_checkpointing_text_encoder:
        print("gradient_checkpointing for text_encoder enabled.")
        model_dict["text_encoder"].language_model.gradient_checkpointing_enable()
    
    if os.path.exists(resume_ckpt_path):
        print("-" * 100)
        state_dict_transformer, state_dict_textencoder = load_model_weights(resume_ckpt_path)
        if len(state_dict_transformer) > 0:
            missing_keys, unexpected_keys = model_dict['pipe'].transformer.load_state_dict(state_dict_transformer, strict=False)
            print("Load {} params for transformer, with {} unexpected_keys".format(len(state_dict_transformer), len(unexpected_keys)))
        if len(state_dict_textencoder) > 0:
            missing_keys, unexpected_keys = model_dict['pipe'].text_encoder.load_state_dict(state_dict_textencoder, strict=False)
            print("Load {} params for text_encoder, with {} unexpected_keys".format(len(state_dict_textencoder), len(unexpected_keys)))
        print("Resume from {} done".format(resume_ckpt_path))
        
    model_dict["pipe"] = model_dict["pipe"].to(device, weight_dtype)   # move to last
    return model_dict

############################################ qwen-image-edit end #########################################


######################################### edit_thinker start #######################################
# 编辑模型的推理流程: thinker -> cot -> edit_model
#   thinker 预测的 cot 替换 edit_model 的原始 prompt

def load_model_weights_thinker_editor(ckpt_path):
    if os.path.isdir(ckpt_path):
        # 切片保存
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
            # 使用 safe_open，它不会立即加载所有张量到内存，而是创建一个映射
            # 这对于内存非常友好
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_name in tensor_names:
                    # 从分片文件中获取单个张量
                    tensor = f.get_tensor(tensor_name)
                    # 放入我们最终的 state_dict 中
                    final_state_dict[tensor_name] = tensor
        state_dict_dit = {k.replace("dit.", ""): v for k, v in final_state_dict.items() if k.startswith("dit.")}
        state_dict_thinker = {k.replace("thinker.", ""): v for k, v in final_state_dict.items() if k.startswith("thinker.")}
        assert len(final_state_dict) == len(state_dict_dit) + len(state_dict_thinker)
    else:
        raise ValueError()
    return state_dict_dit, state_dict_thinker

def load_thinker_editor(edit_model_path, thinker_model_path):    
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        edit_model_path, 
        torch_dtype=torch.bfloat16
    )
    
    if "qwen2.5-vl" in thinker_model_path.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "qwen3-vl" in thinker_model_path.lower():
        if "qwen3-vl-8b" in thinker_model_path.lower():
            model_class = Qwen3VLForConditionalGeneration
        elif "qwen3-vl-30b-a3b" in thinker_model_path.lower():
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"not supported model: {thinker_model_path}")
    else:
        raise ValueError(f"not supported model: {thinker_model_path}")
    
    thinker = model_class.from_pretrained(
            thinker_model_path,
            torch_dtype=torch.float16)
    
    model_dict = {
        "thinker": thinker,

        "pipe": pipe,
        "dit": pipe.transformer,
        # "scheduler": pipe.scheduler,
        # "vae": pipe.vae,
        # "text_encoder": pipe.text_encoder,
    }
    return model_dict

def prepare_thinker_editor(
    model_dict,
    device="cuda",
    weight_dtype=torch.bfloat16,
    gradient_checkpointing_dit=False,
    gradient_checkpointing_thinker=False,
    trainable_module=["thinker", "dit"],
    dit_lora=False,
    thinker_lora=False,
    lora_r: int = 32,
    lora_alpha: int = 32,
    resume_ckpt_path: str = ""
):
    # pipe.vae
    model_dict["pipe"].vae.requires_grad_(False)    
    # pipe.text_encoder
    model_dict["pipe"].text_encoder.requires_grad_(False)

    # pipe.dit
    model_dict["pipe"].transformer.requires_grad_(False)
    if "dit" in trainable_module:
        if dit_lora:
            dit_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.01,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            model_dict["pipe"].transformer.add_adapter(dit_lora_config)
        else:
            model_dict["pipe"].transformer.requires_grad_(True)             # dit full trainable

    if "dit" in trainable_module and gradient_checkpointing_dit:
        print("gradient_checkpointing for dit enabled.")
        model_dict["pipe"].transformer.enable_gradient_checkpointing()
        
    # thinker
    model_dict['thinker'].requires_grad_(False) 
    if "thinker" in trainable_module:
        if thinker_lora:
            thinker_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.01,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model_dict['thinker'].language_model.add_adapter(thinker_lora_config)
        else:
            model_dict["thinker"].requires_grad_(True)             # thinker full trainable
            
    if "thinker" in trainable_module and gradient_checkpointing_thinker:
        print("gradient_checkpointing for thinker enabled.")
        model_dict["thinker"].gradient_checkpointing_enable()
    
    if os.path.exists(resume_ckpt_path):
        print("-" * 100)
        state_dict_dit, state_dict_thinker = load_model_weights_thinker_editor(resume_ckpt_path)
        if len(state_dict_dit) > 0:
            missing_keys, unexpected_keys = model_dict['pipe'].transformer.load_state_dict(state_dict_dit, strict=False)
            print("Dit load {} params for transformer, with {} unexpected_keys".format(len(state_dict_dit), len(unexpected_keys)))
        if len(state_dict_thinker) > 0:
            missing_keys, unexpected_keys = model_dict['thinker'].load_state_dict(state_dict_thinker, strict=False)
            print("Thinker load {} params for text_encoder, with {} unexpected_keys".format(len(state_dict_thinker), len(unexpected_keys)))
        print("Resume from {} done".format(resume_ckpt_path))
        
    model_dict["pipe"] = model_dict["pipe"].to(device, weight_dtype) 
    model_dict["thinker"] = model_dict["thinker"].to(device, weight_dtype)
    return model_dict

######################################### edit-thinker end #########################################
