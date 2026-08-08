#coding=utf-8
# jianchong.zq: for t2i+edit

import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import math
import torch
from typing import Dict, List, Optional, Tuple, Union
# from mdl.distribute_dataset import DistributeDataset as DD
from src.data import utils
import cv2
import re
import os
from torchvision import transforms
import numpy as np
import random
import traceback
import oss2
import tempfile
from PIL import Image

from transformers import AutoProcessor


# 为了兼容新旧版本的 Pillow 库，进行如下处理
try:
    # Pillow 9.0.0 annd later
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    # Old versions
    LANCZOS = Image.ANTIALIAS

def _calculate_resized_dimensions(original_width, original_height, max_area):
    """计算在max_area限制下, 保持长宽比缩放并对齐到32倍数后的新尺寸。"""
    area = original_width * original_height
    if area > max_area:
        scale_factor = math.sqrt(max_area / area)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32
    else:
        new_width = (original_width // 32) * 32
        new_height = (original_height // 32) * 32 
    return new_width, new_height

def preprocess_image(image: Image.Image, max_area = 1024 * 1024, adjust_ar=False) -> Image.Image:
    """
    对给定的图像进行预处理， 保证最终输出的图像 width 和 height 都是 32 的倍数。
    1. 如果面积 > 1024*1024, 则进行下采样。
    2. 如果需要对齐长宽比。
        2.1 中心裁剪到最接近的目标长宽比 (1:1, 16:9, 9:16, 4:3, 3:4)。
    """
    # 1. 调整尺寸 (Resize)
    # --------------------------------------------------------------------------
    original_width, original_height = image.size
    new_width, new_height = _calculate_resized_dimensions(original_width, original_height, max_area)
    
    if (new_width, new_height) != (original_width, original_height):
        image = image.resize((new_width, new_height), resample=LANCZOS)
    
    # 2. 调整长宽比
    if adjust_ar:
        # 2. 中心裁剪 (Center-Crop)
        # --------------------------------------------------------------------------
        # 目标长宽比字典
        TARGET_ASPECT_RATIOS = {
            '1:1': 1.0,
            '16:9': 16 / 9,
            '9:16': 9 / 16,
            '7:5': 7 / 5,
            '5:7': 5 / 7,
            '5:4': 5 / 4,
            '4:5': 4 / 5,
            '4:3': 4 / 3,
            '3:4': 3 / 4,
            '3:2': 3 / 2,
            '2:3': 2 / 3
        }
        current_width, current_height = image.size
        current_ar = current_width / current_height

        best_ar_name = min(
            TARGET_ASPECT_RATIOS.keys(),
            key=lambda name: abs(TARGET_ASPECT_RATIOS[name] - current_ar)
        )
        target_ar = TARGET_ASPECT_RATIOS[best_ar_name]

        # 如果当前图像比目标“更宽”，则以高度为基准计算宽度
        if current_ar > target_ar:
            crop_height = current_height
            crop_width = crop_height * target_ar
        # 如果当前图像比目标“更高”，则以宽度为基准计算高度
        else:
            crop_width = current_width
            crop_height = crop_width / target_ar

        final_width = int(crop_width // 32) * 32
        final_height = int(crop_height // 32) * 32

        # 计算中心裁剪的坐标 (left, top, right, bottom)
        left = (current_width - final_width) / 2
        top = (current_height - final_height) / 2
        right = left + final_width
        bottom = top + final_height
        
        crop_box = (int(left), int(top), int(right), int(bottom))    
        image = image.crop(crop_box)
        
    return image

def statistic_image_pixels(image_list, max_area = 1024 * 1024, adjust_ar=False):
    assert not adjust_ar, "right now noly considering adjust_ar == False"
    total_pixels = 0
    for image in image_list:
        new_width, new_height = _calculate_resized_dimensions(image.width, image.height, max_area)
        total_pixels += new_width * new_height
    return total_pixels

def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'

def oss_download_file(oss_bucket, oss_file, local_file):
    oss_file = oss_file.replace("/data/oss_bucket_0/", "")
    if not oss_bucket.object_exists(oss_file):
        print(f"{oss_file} not exist in oss bucket")
        return False
    else:
        oss_bucket.get_object_to_file(oss_file, local_file)
        return True
    
def spec_prompt_rule_for_text(prompt, replace_prob=0.5):
    """
    针对带文字 prompt 的特殊替换规则 2025-10-01
    1. 将[sep]全部替换为 "" 空字符串
    2. 对于「」特殊括号格式按照一定概率全部替换为“”双引号格式
    3. 不插入split token
    """
    # 基于上面规则修改prompt
    # 1. 将[sep]全部替换为 "" 空字符串
    prompt = prompt.replace("[sep]", "")
    # 2. 对于「」特殊括号格式按照一定概率全部替换为“”双引号格式
    def replace_bookmarks(text, replace_prob):
        def replace_pair(match):
            if random.random() < replace_prob:
                return f'"{match.group(1)}"'
            else:
                return match.group(0)
        pattern = r"「(.*?)」"
        result = re.sub(pattern, replace_pair, text)
        return result
    prompt = replace_bookmarks(prompt, replace_prob=replace_prob)
    # 3. 不插入split token（无需处理）
    return prompt





if __name__ == "__main__":
    import pdb; pdb.set_trace()
    
    utils.logger.info("Dataset preparation completed")

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    utils.logger.info("DataLoader preparation completed")

    for i, data_dict in enumerate(dataloader):
        utils.logger.info(f"Processing batch {i}")
        break
