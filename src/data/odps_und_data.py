#coding=utf-8
# jianchong.zq: 多模态理解数据
# 如果notebook调试，注意先配置 ~/.odps_config.ini

import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

from termcolor import colored
import random
import json 
import torch
from typing import Dict, List, Optional, Tuple, Union
import traceback
import oss2
import tempfile
from copy import deepcopy
from PIL import Image
from functools import partial

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from src.data import utils


def convert_to_qwen_vl_format(dialogue: list, image_paths: list[str]) -> list:
    """
    将指定的对话格式转换为 Qwen-VL 系列模型所需的格式，支持单图和多图。
    Args:
        dialogue (list): 原始对话列表，每个元素是一个包含 'from' 和 'value' 的字典。
        image_paths (list[str]): 一个包含所有图片路径或 URL 的列表。
                                 其顺序必须与对话中 <image> 标签出现的顺序严格一致。
    Returns:
        list: 转换后符合 Qwen-VL 格式的对话列表。
    """
    image_path_iterator = iter(image_paths)
    num_image = 0
    qwen_format_dialogue = []

    for idx, turn in enumerate(dialogue):
        role = ""
        if turn['from'] == 'human':
            role = 'user'
        elif turn['from'] == 'gpt':
            role = 'assistant'
        elif turn['from'] == "system":
            # 处理 system prompt
            assert idx == 0, "system prompt should be the first in dialogue: {}".format(repr(dialogue))
            qwen_format_dialogue.append({"role": "system", "content": turn["value"]})
            continue
        else:
            print(colored(f"Unknown role of dialogue: {turn['from']}", "green", attrs=["bold"]))
            continue

        value = turn['value']
        
        if role == 'user' and '<image>' in value:
            # 这是多模态消息（可能包含多张图片和文本）
            content_list = []
            # 按 <image> 标签分割文本
            text_parts = value.split('<image>')
            
            for i, text_part in enumerate(text_parts):
                # 清理并添加文本部分（如果非空）
                cleaned_text = text_part.strip()
                if cleaned_text:
                    content_list.append({"type": "text", "text": cleaned_text})

                # 在文本部分之后添加图片（除了最后一个文本部分）
                if i < len(text_parts) - 1:
                    image_path = next(image_path_iterator)
                    content_list.append({
                        "type": "image",
                        "image": image_path
                    })
                    num_image += 1
            content = content_list
        else:
            # 这是纯文本消息
            content = value        
        qwen_format_dialogue.append({'role': role, 'content': content})
    
    assert num_image == len(image_paths)
    return qwen_format_dialogue

def oss_download_file(oss_bucket, oss_file, local_file):
    oss_file = oss_file.replace("/data/oss_bucket_0/", "")
    if not oss_bucket.object_exists(oss_file):
        print(f"{oss_file} not exist in oss bucket")
        return False
    else:
        oss_bucket.get_object_to_file(oss_file, local_file)
        return True

class CustomDataProcessor:
    @staticmethod
    def collate_fn(examples, processor):
        """Combine multiple samples into a batch."""    
        # Filter out None samples that might have been returned by __getitem__ on error
        examples = [e for e in examples if e is not None]
        record_ids = [e.pop("record_id", "") for e in examples]
                        
        for e in examples:
            for k in e:
                assert k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"], "k == {}".format(k)
        
        # Find the max length in the batch for padding
        max_length = max([e['input_ids'].size(0) for e in examples])
        pad_token_id = processor.tokenizer.pad_token_id
        
        # 2. Pad 'input_ids', 'attention_mask', and 'labels' to max_length
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for e in examples:
            # Calculate how much padding is needed
            pad_len = max_length - e['input_ids'].size(0)
            
            # Pad input_ids (right padding)
            input_ids_padded = torch.cat([e['input_ids'],  torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
            input_ids_batch.append(input_ids_padded)
            
            # Pad attention_mask (right padding)
            attention_mask_padded = torch.cat([e['attention_mask'], torch.zeros(pad_len, dtype=torch.long)], dim=0)
            attention_mask_batch.append(attention_mask_padded)
            
            # Pad labels (right padding with -100)
            labels_padded = torch.cat([e['labels'], torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
            labels_batch.append(labels_padded)
        
        # Stack the padded tensors to form the final batch
        inputs = {
            "input_ids": torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attention_mask_batch),
            "labels": torch.stack(labels_batch),
        }
        
        # 4. Concatenate vision-related tensors if they exist
        pixel_values = [e["pixel_values"] for e in examples if "pixel_values" in e and e["pixel_values"] is not None]
        if pixel_values:
            inputs["pixel_values"] = torch.cat(pixel_values, dim=0)
        
        image_grid_thw = [e["image_grid_thw"] for e in examples if "image_grid_thw" in e and e["image_grid_thw"] is not None]
        if image_grid_thw:
            inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        
        inputs["record_ids"] = record_ids
        return inputs     
    

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    
    utils.logger.info("Dataset preparation completed")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn,)
    utils.logger.info("DataLoader preparation completed")
    
    for i, data_dict in enumerate(dataloader):
        utils.logger.info(f"Processing batch {i}")
        if i > 100:
            break
    
    
    if False:
        from src.pipe.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
        from peft import LoraConfig
        
        pipe = QwenImageEditPlusPipeline.from_pretrained(
                "/tmp/jianchong.zq/checkpoints/Qwen-Image-Edit-2509/", 
                torch_dtype=torch.bfloat16
            )
        text_encoder = pipe.text_encoder
        text_encoder_lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            lora_dropout=0.01,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        text_encoder.language_model.add_adapter(text_encoder_lora_config)
        
        text_encoder = text_encoder.to("cuda")
        
        def _train_step_und(batch, text_encoder):
            device = text_encoder.device
            weight_dtype = text_encoder.dtype
            
            record_ids = batch.pop("record_ids", None)
            for k in batch:
                batch[k] = batch[k].to(device=device)
                if k not in ["input_ids", "labels", "image_grid_thw"]:
                    batch[k] = batch[k].to(dtype=weight_dtype)    
            outputs = text_encoder(**batch)
            loss = outputs.loss            
            return loss
        
        for i, data_dict in enumerate(dataloader):
            utils.logger.info(f"Processing batch {i}")
            loss = _train_step_und(data_dict, text_encoder)
            if i > 10:
                break
