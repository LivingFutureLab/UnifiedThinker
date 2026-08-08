#coding=utf-8
#jianchong.zq: revise from https://github.com/stepfun-ai/Step1X-Edit/tree/main/GEdit-Bench

import os, sys
from typing import Union, List, Optional
from io import BytesIO
import random
import numpy as np
import base64
import json
#import re
import regex as re  # pip install regex
import math
from PIL import Image
import time

import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


_context_no_delimit = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""

_context = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (the delimiter is necessary. Keep your reasoning concise and short.):
||V^=^V||
{
"score" : 
"reasoning" : 
}
||V^=^V||"""

_context_no_format = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials."""

_prompts_1shot_multi_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: A digital illustration of a cat beside a wooden pot
Output:
||V^=^V||
{
"score" : [5, 5, 10],
"reasoning" :  "The cat is not beside the wooden pot. The pot looks partially resemble to the subject pot. The cat looks highly resemble to the subject cat."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>"""

_prompts_1shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

First lets look at the first set of input (1st image) as an example. 
Text Prompt: A pink and a white frisbee are on the ground.
Output:
||V^=^V||
{
"score" : [5],
"reasoning" :  "White frisbee not present in the image."
}
||V^=^V||

Now evaluate the second set of input (2nd image).
Text Prompt: <prompt>
"""

_prompts_1shot_tie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st, 2nd and 3rd images) as an example. 
Subject: <subject>
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The monster toy looks partially resemble to the token subject. The edit is minimal."
}
||V^=^V||

Now evaluate the second set of input (4th, 5th, and 6th images).
Subject: <subject>
"""

_prompts_1shot_one_image_gen_rule = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: a red cartoon figure eating a banana
Output:
||V^=^V||
{
"score" : [10, 5],
"reasoning" :  "The red cartoon figure is eating a banana. The red cartoon figure looks partially resemble to the subject."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_rule_PQ = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]


First lets look at the first set of input (1st image) as an example. 
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The image gives an unnatural feeling on hands of the girl. There is also minor distortion on the eyes of the girl."
}
||V^=^V||

Now evaluate the second set of input (2nd image).

"""

_prompts_1shot_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_cig_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: the bridge is red, Golden Gate Bridge in San Francisco, USA
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The bridge is red. But half of the bridge is gone."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_two_image_edit_rule = """RULES of each set of inputs:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_1shot_subject_image_edit_rule = """RULES of each set of inputs:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_1shot_control_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_one_video_gen_rule = """RULES:

The images are extracted from a AI-generated video according to the text prompt.
The objective is to evaluate how successfully the video has been generated.
"""

_prompts_0shot_t2v_rule_PQ = """RULES:

The image frames are AI-generated.
The objective is to evaluate how successfully the image frames has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the image frames naturalness. 
(
    0 indicates that the scene in the image frames does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image frames looks natural.
)
A second score from 0 to 10 will rate the image frames artifacts. 
(
    0 indicates that the image frames contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image frames has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

Text Prompt: <prompt>
"""

_prompts_0shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

Subject: <subject>"""

_prompts_0shot_subject_image_edit_rule = """RULES:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_0shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

Text Prompt: <prompt>
"""

_prompts_0shot_tie_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""

_prompts_0shot_cig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

Text Prompt: <prompt>"""

_prompts_0shot_control_image_gen_rule = """RULES:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_t2v_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image frames does not follow the prompt at all. 10 indicates the image frames follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""

_prompts_0shot_multi_subject_image_gen_rule = """RULES:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_subject_image_gen_rule = """RULES:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_one_image_gen_rule = """RULES:

The image is an AI-generated image according to the text prompt.
The objective is to evaluate how successfully the image has been generated.
"""


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QwenVL:
    def __init__(self, prerained_model_path, device_map="auto") -> None:    
        if "qwen2.5-vl" in prerained_model_path.lower():
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "qwen3-vl" in prerained_model_path.lower():
            if "qwen3-vl-8b" in prerained_model_path.lower():
                model_class = Qwen3VLForConditionalGeneration
            elif "qwen3-vl-30b-a3b" in prerained_model_path.lower():
                model_class = Qwen3VLMoeForConditionalGeneration
            else:
                raise ValueError(f"not supported model: {prerained_model_path}")
        else:
            raise ValueError(f"not supported model: {prerained_model_path}")
     
        self.model = model_class.from_pretrained(
            prerained_model_path,
            torch_dtype=torch.float16, 
            device_map=device_map
        ).eval()
        self.processor = AutoProcessor.from_pretrained(prerained_model_path)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        image_links_base64 = []
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_link} for img_link in image_links
                ] + [{"type": "text", "text": text_prompt}]
            }
        ]
        return messages

    def get_parsed_output(self, messages):
        set_seed(42)
        # Prepare the inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate output
        generation_config = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
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
     
class VIEScore:
    def __init__(self, task="tie", pretrained_model="qwen2.5-vl-7b", device_map="auto") -> None:
        self.task = task

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        self.model = QwenVL(pretrained_model, device_map=device_map)
        
        self.context = _context_no_delimit
        if self.task == "t2i":
            self.SC_prompt = "\n".join([self.context, _prompts_0shot_one_image_gen_rule, _prompts_0shot_t2i_rule_SC])
            self.PQ_prompt = "\n".join([self.context, _prompts_0shot_rule_PQ])
        elif self.task == "tie":
            self.SC_prompt = "\n".join([self.context, _prompts_0shot_two_image_edit_rule, _prompts_0shot_tie_rule_SC])
            self.PQ_prompt = "\n".join([self.context, _prompts_0shot_rule_PQ])
        elif self.task == "t2v":
            self.SC_prompt = "\n".join([self.context, _prompts_0shot_one_video_gen_rule, _prompts_0shot_t2v_rule_SC])
            self.PQ_prompt = "\n".join([self.context, _prompts_0shot_t2v_rule_PQ])

    def evaluate(self, image_prompts, text_prompt, extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.task == "t2i":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        elif self.task == "tie":
            _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        elif self.task == "t2v":
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        else:
            raise ValueError()
        
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)
        if self.task == "tie":
            PQ_prompt_final = self.model.prepare_prompt(image_prompts[-1], self.PQ_prompt)
        else:
            PQ_prompt_final = self.model.prepare_prompt(image_prompts, self.PQ_prompt)

        results_dict = {}

        SC_dict = False
        PQ_dict = False
        tries = 0
        max_tries = 1
        while SC_dict is False or PQ_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_SC = self.model.get_parsed_output(SC_prompt_final)
            result_PQ = self.model.get_parsed_output(PQ_prompt_final)
            SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=guess_if_cannot_parse)
            PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=guess_if_cannot_parse)

        if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded") 
            raise ValueError("rate_limit_exceeded")
        results_dict['SC'] = SC_dict
        results_dict['PQ'] = PQ_dict
        if echo_output:
            print("results_dict", results_dict)
        if extract_all_score:
            SC_score = min(results_dict['SC']['score'])
            PQ_score = min(results_dict['PQ']['score'])
            O_score = math.sqrt(SC_score * PQ_score)
            return [SC_score, PQ_score, O_score]
        if extract_overall_score_only:
            SC_scores = results_dict['SC']['score']
            PQ_scores = results_dict['PQ']['score']
            O_score = math.sqrt(min(SC_scores) * min(PQ_scores))
            return O_score
        return results_dict

def calculate_dimensions(target_area, ratio):
    """
    根据给定的目标面积和宽高比，计算新的宽度和高度。
    这个函数确保调整大小后的图像面积接近目标面积，同时保持原始的形状。

    Args:
        target_area (float): 目标图像面积 (例如: 512 * 512)。
        ratio (float): 原始图像的宽高比 (width / height)。

    Returns:
        tuple: (计算出的宽度, 计算出的高度, 计算出的新面积)，均为整数。
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)

def process_single_item(item, vie_score, max_retries=10):
    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    intersection_exist = item['Intersection_exist']
    edit_image_path = item['edited_image_path']
    
    for retry in range(max_retries):
        try:
            pil_image =  Image.open(item['input_image_raw']).convert("RGB")
            pil_image_edited = Image.open(edit_image_path).convert("RGB")
            
            # 调整图像尺寸
            source_img_width, source_img_height, source_img_area = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
            edited_img_width, edited_img_height, edited_img_area = calculate_dimensions(512 * 512, pil_image_edited.width / pil_image_edited.height)
            pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
            pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))
            
            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image, pil_image_edited], text_prompt)
            sementics_score, quality_score, overall_score = score_list

            print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
            
            return {
                "key": key,
                "edited_image": edit_image_path,
                "instruction": instruction,
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "overall_score": overall_score,
                "intersection_exist" : item['Intersection_exist'],
                "instruction_language" : item['instruction_language']
            }
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                print(f"Error item: {item}")
                time.sleep(wait_time)
            else:
                print(f"Failed to process after {max_retries} attempts: {e}")
                return None
            
            
if __name__ == "__main__":
    vie_scorer = VIEScore(task='tie', pretrained_model="/tmp/jianchong.zq/checkpoints/Qwen3-VL-8B-Instruct/", device_map="auto")
        
    # from datasets import load_dataset, load_from_disk
    # #dataset = load_dataset("stepfun-ai/GEdit-Bench")
    
    datas = json.load(open("/data/oss_bucket_0/mllm_dataset/public_datasets/image_edit/GEdit-Bench/gedit_1k.json"))
    edited_images_dir = "/data/oss_bucket_0/mllm_dataset/public_datasets/image_edit/GEdit-Bench/images/"
    
    for idx in range(5):
        data = datas[idx]
        data["edited_image_path"] = os.path.join(edited_images_dir, data["input_image_raw"])
        data["input_image_raw"] = os.path.join(edited_images_dir, data["input_image_raw"])
           
        result = process_single_item(data, vie_scorer)
        print(result)
