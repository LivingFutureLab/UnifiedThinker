#coding=utf-8
# jianchong.zq, load from json file

import torch
from typing import Dict, List, Optional, Tuple, Union
from src.data import utils
import cv2
import os
from torchvision import transforms
import numpy as np
import random
import traceback
from PIL import Image
from termcolor import colored

import oss2
import json
import tempfile
from torch.utils.data import Dataset

from src.data.odps_t2i_edit_data import (oss_download_file, spec_prompt_rule_for_text, statistic_image_pixels,
                                         preprocess_image, get_caption_language)


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
    
class LocalT2iEditDataset(Dataset):
    def __init__(
        self,
        data_files,
        data_weights = None,
        text_drop: float = 0.1,
        oss_access_id="",
        oss_access_key="",
        bucket_name="",
        oss_endpoint="",
        tgt_image_max_area: int = 1024 * 1024,              # t2i/edit 目标图像的最大像素数目
        cond_image_max_area: int = 1024 * 1024,             # edit 条件图的最大像素数目
        multi_cond_image_max_area: int = 1024 * 1024 * 2,
        adjust_ar: bool = False,
        replace_prompt_with_cot: bool = False               # for thinker_editor
    ):
        super(LocalT2iEditDataset, self).__init__()
        self.bucket_name = bucket_name
        self.oss_bucket = ""
        
        self.tgt_image_max_area = tgt_image_max_area
        self.cond_image_max_area = cond_image_max_area
        self.multi_cond_image_max_area = multi_cond_image_max_area
        self.adjust_ar = adjust_ar
        
        self.replace_prompt_with_cot = replace_prompt_with_cot  # for thinker_editor model, 直接使用 cot 替换原始 prompt
        
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
            
        self.text_drop = text_drop
                
        random.seed(0)
        random.shuffle(self.datas)
    
    def __len__(self):
        return len(self.datas)
    

    def _download_and_open_image(self, path: str) -> Optional[Image.Image]:
        try:
            if not isinstance(path, str):
                return None

            if not os.path.exists(path):
                utils.logger.warning(f"Image path not found: {path}")
                return None

            return Image.open(path).convert("RGB")

        except Exception as e:
            utils.logger.warning(f"Failed to download/open image {path}: {e}")
            return None



    def __getitem__(self, idx):
        """Process a single data sample."""
        sample = self.datas[idx]
        try:
            ## prompt
            prompt_list = []
            prompt_list_prob = []
            allowed_keys = {'cn_long', 'cn_short', 'en_long', 'en_short'}
            try:
                text_inst = eval(sample["text_inst"])
                filtered_text_inst = {key: value for key, value in text_inst.items() if key in allowed_keys}
                prompt_list += list(filtered_text_inst.values())
                prompt_list_prob += [0.8] * len(filtered_text_inst) # text_inst 的采样概率调整为 0.8
            except Exception as e:
                text_inst = None 
            try:
                text_dec = eval(sample["text_dec"])
                filtered_text_dec = {key: value for key, value in text_dec.items() if key in allowed_keys}
                prompt_list += list(filtered_text_dec.values())
                prompt_list_prob += [0.2] * len(filtered_text_dec)  # text_dec 的采样概率调整为 0.2
            except Exception as e:
                text_dec = None
            prompt = random.choices(prompt_list, weights=prompt_list_prob, k=1)[0]
            prompt = spec_prompt_rule_for_text(prompt)
            
            
            ## prompt_cot
            allowed_cot_keys = {'cn_cot', 'en_cot'}
            prompt_cot_list = []
            try:
                text_inst = eval(sample["text_inst"])
                filtered_text_inst = {key: value for key, value in text_inst.items() if key in allowed_cot_keys}
                prompt_cot_list += list(filtered_text_inst.values())
            except Exception as e:
                pass
            try:
                text_dec = eval(sample["text_dec"])
                filtered_text_dec = {key: value for key, value in text_dec.items() if key in allowed_cot_keys}
                prompt_cot_list += list(filtered_text_dec.values())
            except Exception as e:
                pass
            if len(prompt_cot_list) > 0:
                prompt_cot = random.choice(prompt_cot_list)
            else:
                prompt_cot = None    
            
            ## condition images
            try:
                ref_imgs_oss_path = eval(sample["ref_imgs_oss_path"])
            except Exception as e:
                ref_imgs_oss_path = []
                
            raw_condition_images = []
            for ref_image_file in ref_imgs_oss_path: 
                ref_image = self._download_and_open_image(ref_image_file)
                raw_condition_images.append(ref_image)
            
            # step-1: 计算 cond_image_max_area, 保证所有条件图的总的pixels <= multi_cond_image_max_area
            cond_image_max_area = self.cond_image_max_area 
            while True:
                total_pixels = statistic_image_pixels(raw_condition_images, cond_image_max_area, adjust_ar=self.adjust_ar)
                if total_pixels <= self.multi_cond_image_max_area:
                    break
                cond_image_max_area = cond_image_max_area // 2
            # step-2: resize ref image
            raw_condition_images = [preprocess_image(ref_image, max_area=cond_image_max_area, adjust_ar=self.adjust_ar) for ref_image in raw_condition_images]
                
                 
            ## target image
            tgt_imgs_oss_path = eval(sample["tgt_imgs_oss_path"])
            assert len(tgt_imgs_oss_path) == 1, "right now support single image generation"
            
            image_file = tgt_imgs_oss_path[0]
            image = self._download_and_open_image(image_file)
            # resize image
            image = preprocess_image(image, max_area=self.tgt_image_max_area, adjust_ar=self.adjust_ar)
            
            if self.replace_prompt_with_cot and prompt_cot is not None:
                if idx % 100 == 0:
                    print(colored("replace_prompt_with_cot: {} -> {}".format(prompt, prompt_cot), "green", attrs=["bold"]))
                prompt = prompt_cot
                prompt_cot = None
                
            if len(raw_condition_images) == 0:
                # t2i task
                # following qwen-image
                positive_magic = {
                    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
                    "zh": ", 超清，4K，电影级构图." # for chinese prompt
                }
                prompt = prompt + positive_magic[get_caption_language(prompt)]
                            
            if random.random() <= self.text_drop:
                prompt = ""
                if prompt_cot is not None:
                    prompt_cot = None
                
            result = {
                "raw_condition_images": raw_condition_images,
                "image": image,
                "prompt": prompt,
                "prompt_cot": prompt_cot
            }
            if idx % 100 == 0:
                print(f"[Dataset-t2i-edit] [image]: {image.size}, [raw_condition_images]: {len(raw_condition_images)}, [prompt_cot]: {repr(prompt_cot) if prompt_cot is not None else None}, [prompt]: {repr(prompt)}\n")
            return result

        except Exception as e:
            utils.logger.error("dataload 样本报错: {}".format(str(e)))
            return {
                    "raw_condition_images": [],
                    "image": Image.new('RGB', (1024, 1024), (255, 255, 255)), # 白色图片
                    "prompt": "a white image.",
                    "prompt_cot": None}

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Dict:
        """Combine multiple samples into a batch."""
        batch = {}
        for k, v in examples[0].items():
            cur_batch_record = [example[k] for example in examples]
            batch[k] = cur_batch_record
        return batch
