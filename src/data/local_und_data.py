#coding=utf-8
# jianchong.zq, demo understanding dataset for debug only.

import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    
import io
import json
import random
import torch
from PIL import Image
import oss2
import tempfile
import traceback
from functools import partial
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from torch.utils.data import Dataset
from termcolor import colored

from src.data import utils
from src.data.odps_und_data import convert_to_qwen_vl_format, oss_download_file, CustomDataProcessor


Image.MAX_IMAGE_PIXELS = 20_000_000


def read_json_data(file_name):
    if file_name.endswith(".json"):
        with open(file_name, 'r') as f:
            datas = json.load(f)
        return datas
    elif file_name.endswith(".jsonl"):
        datas = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == "": continue
                datas.append(json.loads(line))
        return datas
    else:
        raise ValueError("not supported file: {}".format(file_name))
    
class LocalUndDataset(Dataset):
    def __init__(
        self, 
        data_files, 
        data_weights=None,
        qwenvl_pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_sequent_length: int = 4096,
        oss_access_id="",
        oss_access_key="",
        bucket_name="",
        oss_endpoint="",
    ):
        super(LocalUndDataset, self).__init__()
        self.bucket_name = bucket_name
        self.oss_bucket =""
        
        self.datas = []    
        if isinstance(data_files, str):
            data_files = [data_files]
        
        if data_weights is None:
            data_weights = [1.0] * len(data_files)
        else:
            assert len(data_weights) == len(data_files)
            
        for data_file, data_weight in zip(data_files, data_weights):
            tmp_datas = None
            if not os.path.exists(data_file):
                # download from oss 
                object_key = data_file.replace(f"oss://{self.bucket_name}/", "")
                suffix = os.path.splitext(object_key)[1]

            else:
                # with open(data_file, 'r') as f:
                #     tmp_datas = json.load(f)
                tmp_datas = read_json_data(data_file)
            
            if tmp_datas is not None:
                tmp_datas = tmp_datas * int(data_weight)    # 重采样
                print(colored("[Dataset] {} samples read from: {}".format(len(tmp_datas), data_file), "green", attrs=["bold"]))
                
                self.datas += tmp_datas
                
        print(colored("[Dataset] total {} samples".format(len(self.datas)), "green", attrs=["bold"]))
        
        self.vlm_processor = AutoProcessor.from_pretrained(
            qwenvl_pretrained,
            min_pixels=64 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )                
        self.max_sequent_length = max_sequent_length
        
        # Pre-tokenize templates to find assistant responses
        # Note: The exact string depends on the chat template. For Qwen, it includes a newline.
        self.assistant_prompt_ids = self.vlm_processor.tokenizer.encode(
            "\n<|im_start|>assistant\n", add_special_tokens=False
        )
        self.im_end_id = self.vlm_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")        
        self.collate_fn = partial(CustomDataProcessor.collate_fn, processor=self.vlm_processor)
        
        self.dummy_inputs = None    # 用于在dataloader异常时
        
        random.seed(0)
        random.shuffle(self.datas)
    
    def __len__(self):
        return len(self.datas)
    
    def _download_and_open_image(self, path: str) -> Optional[Image.Image]:
        """Open an image from a local path (no OSS)."""
        try:
            if not isinstance(path, str) or path == "":
                return None

            if not os.path.exists(path):
                utils.logger.warning(f"Image path not found: {path}")
                return None

            return Image.open(path).convert("RGB")

        except Exception as e:
            utils.logger.warning(f"Failed to open image {path}: {e}")
            return None

                    
    def __getitem__(self, idx):
        data = self.datas[idx]
        try:            
            record_id = data["record_id"]
            llava_data = data["llava_data"]
            num_images = int(data["num_images"])
            if num_images > 0:
                image_list = data["image"].split("<|path_separator|>")
                assert len(image_list) == num_images, "num_images is wrong for record_id: {}".format(record_id)
            else:
                image_list = []
            
            llava_data = json.loads(llava_data)
            conversations_raw = deepcopy(llava_data["conversations"])
            conversations = convert_to_qwen_vl_format(conversations_raw, image_list)
            
            for msg in conversations:
                if isinstance(msg["content"], str):
                    continue
                for cnt in msg["content"]:
                    if cnt["type"] == "image":                    
                        img = self._download_and_open_image(cnt["image"])
                        cnt["image"] = img
            
            text = self.vlm_processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(conversations)
            

            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=False,       # Padding will be handled by the collate_fn
                max_length=self.max_sequent_length, 
                truncation=True,    # 当序列超过最大长度时进行截断
            )      
            input_ids = inputs["input_ids"][0]
            # Create labels tensor: mask out everything except assistant's responses
            labels = torch.full_like(input_ids, -100)
            
            # Find all occurrences of the assistant prompt
            input_ids_list = input_ids.tolist()
            assistant_prompt_len = len(self.assistant_prompt_ids)
            
            i = 0
            while i < len(input_ids_list):
                # Check if the sequence from index `i` matches the assistant prompt
                if input_ids_list[i:i + assistant_prompt_len] == self.assistant_prompt_ids:
                    # Found an assistant response start
                    response_start_idx = i + assistant_prompt_len
                    # Find the end of the response, marked by <|im_end|>
                    try:
                        response_end_idx = input_ids_list.index(self.im_end_id, response_start_idx)
                    except ValueError:
                        # If <|im_end|> is not found, assume the response goes to the end
                        response_end_idx = len(input_ids_list)
                    
                    # Copy the token IDs of the response to the labels tensor
                    labels[response_start_idx:response_end_idx] = input_ids[response_start_idx:response_end_idx]
                    
                    # Move index `i` past this found response
                    i = response_end_idx
                else:
                    i += 1

            # Also mask out special visual tokens in labels, although they are likely
            # already -100 because they are not part of an assistant response.
            visual_tokens_ids = [
                self.vlm_processor.tokenizer.convert_tokens_to_ids(self.vlm_processor.image_token),
                self.vlm_processor.tokenizer.convert_tokens_to_ids(self.vlm_processor.video_token)
            ]
            for token_id in visual_tokens_ids:
                if token_id is not None:
                    labels[labels == token_id] = -100
            
            # Add the carefully crafted labels to the output
            inputs['labels'] = labels.unsqueeze(0) # Add batch dimension
            
            # Squeeze out the batch dimension added by the vlm_processor
            for k in inputs:
                if inputs[k] is not None and isinstance(inputs[k], torch.Tensor):
                    if k not in ["pixel_values", "image_grid_thw"]: # These are already batched correctly
                        inputs[k] = inputs[k].squeeze(0)     
                        
            if idx % 100 == 0:
                if "pixel_values" in inputs:
                    print(f"[Dataset-und] [record_id]: {record_id}, [pixel_values]: {inputs['pixel_values'].shape}, [text]: {repr(text)}")  
                else:
                    print(f"[Dataset-und] [record_id]: {record_id}, [text]: {repr(text)}")  
                                 
            if self.dummy_inputs is None:
                self.dummy_inputs = inputs
            inputs["record_id"] = record_id
            return inputs
            
        except Exception as e:
            utils.logger.error(
                f"Error processing sample: {str(e)}\n{traceback.format_exc()}"
            )
            utils.logger.error("[dataset-und] 报错")
            return self.dummy_inputs


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    utils.logger.info("Dataset preparation completed")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn)
    utils.logger.info("DataLoader preparation completed")
    
    for i, data_dict in enumerate(dataloader):
        utils.logger.info(f"Processing batch {i}")
        if i > 100:
            break
    