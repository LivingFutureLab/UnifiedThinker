import random
from io import BytesIO
from logging import getLogger
import json
import os
import re
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoProcessor

logger = getLogger(__name__)

OSS_PUBLIC_ROOT = "/data/oss_bucket_0"


def down_pth_feat(oss_path_real_image):
    if type(oss_path_real_image) is str:
        loaded_tensor = torch.load(
            os.path.join(OSS_PUBLIC_ROOT, oss_path_real_image), map_location="cpu"
        )
    else:
        loading_stream = BytesIO(oss_path_real_image)
        loaded_tensor = torch.load(loading_stream, map_location="cpu")
    loaded_tensor = loaded_tensor.detach().squeeze(0)
    return loaded_tensor


def get_json(json_string):
    return json.loads(json_string)


def find_nearest16(value: int) -> int:
    """Round a number to the nearest multiple of 16."""
    return int(value / 16 + 0.5) * 16


def calculate_resize_scale_by_area(
    img_size: Tuple[int, int], target_area: int
) -> Tuple[int, int, float]:
    """
    Calculate new dimensions while maintaining aspect ratio to match target area.

    Args:
        img_size: Original image dimensions (height, width)
        target_area: Target area in pixels

    Returns:
        Tuple of (new_height, new_width, scale_factor)
    """
    ori_h, ori_w = img_size
    ori_area = ori_h * ori_w
    scale = (target_area / ori_area) ** 0.5
    new_h = find_nearest16(round(scale * ori_h))
    new_w = find_nearest16(round(scale * ori_w))
    return new_h, new_w, scale


def calculate_new_h_new_w_by_16(img_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate new dimensions while maintaining aspect ratio to match target area.

    Args:
        img_size: Original image dimensions (height, width)

    Returns:
        Tuple of (new_height, new_width)
    """
    ori_h, ori_w = img_size
    new_h = find_nearest16(ori_h)
    new_w = find_nearest16(ori_w)
    return new_h, new_w


def get_position_ids(
    prompt: str, tokenizer: AutoProcessor, max_token_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get position IDs for Qwen model input.

    Args:
        prompt: Input text prompt
        tokenizer: Qwen tokenizer instance

    Returns:
        Tuple of (qwen_position_ids, visual_position_ids)
    """
    prompt = [prompt]
    prompt_qwen_input = []

    for prompt_i in prompt:
        instructions = {
            "Text input": prompt_i,
            "Instruction editing description": "no",
            "image input": "no",
        }
        message = [
            {"role": "user", "content": [{"type": "text", "text": str(instructions)}]}
        ]
        prompt_qwen_input.append(
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        )

    inputs = tokenizer(
        text=prompt_qwen_input,
        images=None,
        videos=None,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = inputs["offset_mapping"][0]
    start_pos = prompt_qwen_input[0].find(prompt[0])
    end_pos = start_pos + len(prompt[0])
    if start_pos == -1:
        escaped_prompt = str({"p": prompt[0]})[7:-2]
        # print(f"excapted prompt: {escaped_prompt}")
        start_pos = prompt_qwen_input[0].find(escaped_prompt)
        end_pos = start_pos + len(escaped_prompt)
        assert start_pos != -1

    prompt_qwen_input_token_ids = np.zeros(len(prompt_qwen_input[0]))
    for i, (pos_id_start, pos_id_end) in enumerate(offset_mapping):
        prompt_qwen_input_token_ids[pos_id_start:pos_id_end] = i

    qwen_pos_ids = torch.arange(offset_mapping.shape[0])
    visual_pos_ids = torch.tensor(prompt_qwen_input_token_ids[start_pos:end_pos])
    return qwen_pos_ids, visual_pos_ids


def get_positional_encoding(position: np.ndarray, channels: int) -> torch.Tensor:
    """Generate positional encoding using sine and cosine functions."""
    position = position.reshape(-1, 1)
    div_term = np.exp(np.arange(0, channels, 2) * -(np.log(10000.0) / channels))
    pe = np.zeros((position.shape[0], channels))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe)


def query_glyph_feat(feat_dict: Dict[str, torch.Tensor], prompt: str) -> torch.Tensor:
    """Extract glyph features for each character in the prompt."""
    feat_list = []
    for char in prompt:
        if char not in feat_dict:
            logger.warning(f"Character {char} not found in feat dict")
            feat_dim = list(feat_dict.values())[0].shape[-1]
            feat_list.append(torch.zeros(feat_dim))
        else:
            feat_list.append(feat_dict[char])
    return torch.stack(feat_list)


def replace_oss_prefix(path_string):
    """
    替换给定的字符串路径前缀。
    - 'oss://alimama-creative-public' 替换为 '/data/oss_bucket_1'
    - 'oss://alimama-creative' 替换为 '/data/oss_bucket_0'
    """
    if path_string.startswith("oss://alimama-creative-public"):
        return path_string.replace(
            "oss://alimama-creative-public", "/data/oss_bucket_1", 1
        )
    elif path_string.startswith("oss://alimama-creative"):
        return path_string.replace("oss://alimama-creative", "/data/oss_bucket_0", 1)
    else:
        return path_string


def extract_quoted_text_indexes(text, return_raw_info=False):
    pattern = r"「(.*?)」"
    results = []

    index_mask = np.zeros(len(text))

    for match in re.finditer(pattern, text):
        content = match.group(1)
        start_index = match.start(1)  # 内容起始索引
        end_index = match.end(1) - 1  # 内容结束索引

        index_mask[start_index : end_index + 1] = 1

        # 检查内容中是否包含[sep]分隔符
        sep_start = content.find("[sep]")
        has_sep = sep_start != -1

        sep_info = None
        if has_sep:
            # 计算[sep]在原始文本中的位置
            sep_start_index = start_index + sep_start
            sep_end_index = sep_start_index + 4  # [sep]占5个字符，索引从0-4

            # 记录分隔符前后的内容
            before_sep = content[:sep_start]
            after_sep = content[sep_start + 5 :]

            sep_info = {
                "sep_start_index": sep_start_index,
                "sep_end_index": sep_end_index,
                "before_sep": before_sep,
                "after_sep": after_sep,
            }

            index_mask[sep_start_index : sep_end_index + 1] = 0

        results.append(
            {
                "content": content,
                "start_index": start_index,
                "end_index": end_index,
                "has_sep": has_sep,
                "sep_info": sep_info,
            }
        )

    if return_raw_info:
        return index_mask, results
    else:
        return index_mask

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]