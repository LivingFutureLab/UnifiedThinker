#coding=utf-8
# Reward model for t2i prism_bench, revise from https://github.com/rongyaofang/prism-bench

import os
import re 
import json
import gc
from PIL import Image
import demjson3
from typing import Any, Dict, List

import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_category_descriptions(caption_path) -> List[str]:
    """
    Loads category descriptions from a .jsonl file.
    """
    descriptions = []
    if not caption_path.exists():
        print(f"Warning: Caption file not found at {caption_path}")
        return descriptions
    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "prompt" in data:
                    descriptions.append(data["prompt"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line in caption file due to error: {e}")
    return descriptions

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

class QwenVL:
    def __init__(self, model_path="", max_new_tokens=1024, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
                 device_map="balanced"):
        if "qwen2.5-vl" in model_path.lower():
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "qwen3-vl" in model_path.lower():
            if "qwen3-vl-8b" in model_path.lower():
                model_class = Qwen3VLForConditionalGeneration
            elif "qwen3-vl-30b-a3b" in model_path.lower():
                model_class = Qwen3VLMoeForConditionalGeneration
            else:
                raise ValueError(f"not supported model: {model_path}")
        else:
            raise ValueError(f"not supported model: {model_path}")
        
        self.model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
        }

    def parse_input(self, query=None, imgs=None):
        if imgs is None:
            messages = [{"role": "user", "content": query}]
            return messages

        if isinstance(imgs, str):
            imgs = [imgs]
        content = []
        for img in imgs:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        return messages

    def chat(self, query=None, imgs=None, history=None):
        if history is None:
            history = []

        user_query = self.parse_input(query, imgs)
        history.extend(user_query)

        text = self.processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                                  add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(history)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda").to(torch.float16)
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        # print(len(generated_ids[0]) - len(inputs.input_ids[0]))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        history.append({"role": "assistant", "content": response})

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        return response, history

def get_model_response(model, prompt: str, image: Image.Image) -> str:
    """
    Sends a request to the Qwen-VL model and returns the response.
    Args:
        model (Qwen25VL): The loaded Qwen-VL model instance.
        prompt (str): The text prompt for evaluation.
        image (Image.Image): The PIL image to be evaluated.

    Returns:
        str: The content of the model's response.
    """
    try:
        response, _ = model.chat(query=prompt, imgs=[image])
        print("Output: ", response)
        return response
    except Exception as e:
        print(f"An error occurred during the model call: {e}")
        return ""

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

def process_single_item(model, image_path, text_prompt, eval_category):
    messages_pool = get_message_templates()
    results = []
    for eval_type in ["alignment", "aesthetic"]:
        if eval_type == "alignment":
            messages_template = messages_pool.get("alignment", {}).get(eval_category)
        else: # aesthetic
            messages_template = messages_pool.get("aesthetic")

        image = Image.open(image_path).convert("RGB")

        final_prompt = messages_template.format(text_prompt=text_prompt)
        output_text = get_model_response(model, final_prompt, image)
        result_data = clean_and_parse_json(output_text)
        results.append(result_data['score'])

    score_align, score_aes = results
    final_score = (score_align + score_aes) / 2.0
    return final_score
    
    
if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    model = QwenVL(model_path="/tmp/jianchong.zq/checkpoints/Qwen3-VL-8B-Instruct", device_map="cuda:0")

    image_path = "/data/oss_bucket_1/jianchong.zq/tbstar_image_eval_results/Qwen-Image-Edit-2509/prism_bench/en/affection/0.png"
    text_prompt = "Amidst the golden glow of a sunrise-drenched cityscape, the morning rush hour unfolds as a symphony of movement and solitude, where the vibrant red buses and cloaked figures navigate the transient dance of urban life, evoking both the anonymity of the crowd and the quiet introspection of individual journeys."
    eval_category = "affection"
    
    score = process_single_item(model, image_path, text_prompt, eval_category)
    print(f"score: {score}")
    