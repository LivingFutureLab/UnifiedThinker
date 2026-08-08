import argparse
import base64
import io
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import demjson3
from openai import OpenAI
from PIL import Image


class ApiConfig:
    """
    A class to store API configuration settings.

    This class is used to manage the API key, base URL, and model name for
    accessing the OpenAI API. By encapsulating these settings, it becomes
    easier to manage different configurations.
    """

    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        """
        Initializes the ApiConfig object.

        Args:
            model (str): The name of the model to use for API calls.
            api_key (str, optional): The API key for authentication.
            base_url (str, optional): The base URL of the API endpoint.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

def parse_arguments():
    """
    Parses command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate text-to-image generation models using an AI evaluator."
    )

    # Required arguments
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Base path to the image directory.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key. It's recommended to use environment variables instead.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="OpenAI base URL for custom or proxy endpoints.",
    )

    # Optional arguments
    parser.add_argument(
        "--api_model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="Name of the OpenAI model to use for evaluation.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to process in the category.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save evaluation scores. Defaults to '[image_path]/score'.",
    )
    parser.add_argument(
        "--zh",
        action="store_true",
        help="Set to process Chinese language content. Changes path from 'en' to 'zh'.",
    )
    
    parser.add_argument('--category', type=str, nargs="+")
    parser.add_argument(
        "--pdb_debug", action="store_true"
    )

    return parser.parse_args()

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encodes a PIL Image object to a Base64 string.

    This function takes an image, saves it to an in-memory buffer, and then
    encodes it to a Base64 string, which is suitable for embedding in API
    requests.

    Args:
        image (Image.Image): The PIL Image object to encode.

    Returns:
        str: The Base64 encoded image as a string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_model_response(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int = 42,
) -> str:
    """
    Sends a request to the OpenAI API and returns the model's response.

    This function constructs the payload for the API call, sends the request,
    and processes the response to extract the content. It also handles model-
    specific payload requirements, such as the 'seed' parameter.

    Args:
        client (OpenAI): The OpenAI client instance.
        messages (List[Dict[str, Any]]): The list of messages for the chat.
        model (str): The model to use for the completion.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        top_p (float): The nucleus sampling probability.
        seed (int, optional): The random seed for reproducibility.

    Returns:
        str: The content of the model's response.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if "gemini" not in model:
        payload["seed"] = seed

    try:
        completion = client.chat.completions.create(**payload)
        print(f"Total tokens used: {completion.usage.total_tokens}")
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return ""


def clean_and_parse_json(json_str: str) -> Dict[str, Any]:
    """
    Cleans and parses a JSON string, with fallback to demjson3.

    This function first attempts to clean the string by removing common non-JSON
    elements like markdown formatting. It then tries to parse the cleaned string
    as JSON. If that fails, it uses the more lenient demjson3 library.

    Args:
        json_str (str): The JSON string to parse.

    Returns:
        Dict[str, Any]: The parsed JSON object.
    """
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    # Remove trailing commas that can cause issues with standard JSON parsers
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


def load_category_descriptions(caption_path: Path) -> List[str]:
    """
    Loads category descriptions from a .jsonl file.

    Args:
        caption_path (Path): The path to the .jsonl file.

    Returns:
        List[str]: A list of caption texts.
    """
    descriptions = []
    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "prompt" in data:
                    descriptions.append(data["prompt"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line due to error: {e}")
    return descriptions

def summarize_results(base_score_dir: Path, eval_pools: List[str], num_images: int):
    """
    Summarizes evaluation results, calculates various average scores,
    and saves the summary to a text file.

    This function calculates:
    1. The average score of each sub-part (alignment and aesthetic).
    2. The overall average score of alignment.
    3. The overall average score of aesthetic.
    4. The combined average score for each sub-part.
    5. The final total average score.

    Args:
        base_score_dir (Path): The base directory where score files are saved.
                               This directory should contain 'alignment' and 'aesthetic' subdirectories.
        eval_pools (List[str]): A list of the evaluation categories (e.g., "affection", "composition").
        num_images (int): The number of images processed per category.
    """
    print("Starting result summarization...")

    scores = {
        "alignment": {category: [] for category in eval_pools},
        "aesthetic": {category: [] for category in eval_pools},
    }

    # Step 1: Load all scores from JSON files
    for eval_type in ["alignment", "aesthetic"]:
        for category in eval_pools:
            for i in range(num_images):
                score_file = base_score_dir / eval_type / category / f"{i}.jsonl"
                if not score_file.exists():
                    continue
                
                try:
                    with open(score_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'score' in data and isinstance(data['score'], (int, float)):
                            scores[eval_type][category].append(data['score'])
                        else:
                            print(f"Warning: 'score' key not found or invalid in {score_file}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not read or parse {score_file}. Error: {e}")

    # Step 2: Calculate averages
    results = {
        "alignment_by_category": {},
        "aesthetic_by_category": {},
        "combined_by_category": {},
        "overall_alignment": 0,
        "overall_aesthetic": 0,
        "final_total_average": 0,
    }
    
    # Helper function for safe averaging
    def calculate_average(score_list: List[float]) -> float:
        if not score_list:
            return 0.0
        return sum(score_list) / len(score_list)

    all_alignment_scores = []
    all_aesthetic_scores = []

    # (1) The average score of each sub-part (for alignment and aesthetic)
    for category in eval_pools:
        alignment_scores = scores["alignment"][category]
        aesthetic_scores = scores["aesthetic"][category]
        
        results["alignment_by_category"][category] = calculate_average(alignment_scores)
        results["aesthetic_by_category"][category] = calculate_average(aesthetic_scores)
        
        all_alignment_scores.extend(alignment_scores)
        all_aesthetic_scores.extend(aesthetic_scores)

        # (4) The average score of alignment and aesthetic combined for each sub-part.
        combined_scores = alignment_scores + aesthetic_scores
        results["combined_by_category"][category] = calculate_average(combined_scores)

    # (2) The average score of alignment.
    results["overall_alignment"] = calculate_average(all_alignment_scores)
    
    # (3) The average score of aesthetic.
    results["overall_aesthetic"] = calculate_average(all_aesthetic_scores)

    # (5) The final average score.
    total_scores = all_alignment_scores + all_aesthetic_scores
    results["final_total_average"] = calculate_average(total_scores)

    # Step 3: Format the output string
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("          EVALUATION SUMMARY REPORT")
    summary_lines.append("="*60)
    
    summary_lines.append("\n--- (1 & 4) Average Scores by Sub-Part ---\n")
    for category in eval_pools:
        summary_lines.append(f"Category: {category.capitalize()}")
        summary_lines.append(f"  - Alignment Average: {results['alignment_by_category'][category]:.2f}")
        summary_lines.append(f"  - Aesthetic Average: {results['aesthetic_by_category'][category]:.2f}")
        summary_lines.append(f"  - Combined Average:  {results['combined_by_category'][category]:.2f}\n")

    summary_lines.append("\n" + "-"*60 + "\n")
    summary_lines.append("--- Overall Average Scores ---\n")
    summary_lines.append(f"(2) Overall Alignment Average: {results['overall_alignment']:.2f}")
    summary_lines.append(f"(3) Overall Aesthetic Average: {results['overall_aesthetic']:.2f}")
    
    summary_lines.append("\n" + "="*60 + "\n")
    summary_lines.append(f"(5) FINAL COMBINED AVERAGE SCORE: {results['final_total_average']:.2f}")
    summary_lines.append("="*60)

    summary_text = "\n".join(summary_lines)

    # Step 4: Print to console and save to file
    print("\n\n" + summary_text)

    output_file_path = base_score_dir / "evaluation_summary.txt"
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\nSummary successfully saved to: {output_file_path}")
    except IOError as e:
        print(f"\nError: Could not write summary to file at {output_file_path}. Error: {e}")

def get_message_templates() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a dictionary of message templates for different evaluation tracks.

    Each template provides instructions to the AI model for a specific
    evaluation task. This centralizes the prompt definitions, making them
    easier to manage and modify.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary where keys are track names
        and values are the corresponding message templates.
    """
    # The message templates are long, so they are kept here as originally
    # defined. In a larger project, you might consider moving these to a
    # separate configuration file (e.g., a YAML or JSON file).
    messages_1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_3 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_4 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_5 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_6 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_7 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
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
""",
                }
            ],
        },
    ]

    messages_8 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text":
""" 
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
        }]
        }
    ]

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


def main():
    """
    Main function to run the image evaluation script.
    """
    args = parse_arguments()
    if args.pdb_debug:
        import pdb; pdb.set_trace()

    if args.category is not None:
        eval_pools = args.category
    else:
        eval_pools = ["imagination", "entity", "text_rendering", "style", "affection", "composition", "long_text"]
    language_code = "zh" if args.zh else "en"
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.output_dir:
        score_root_dir = args.output_dir
    else:
        # Corrected from args.dataset_path to args.image_path
        score_root_dir = args.image_path / "score"

    # --- Start Alignment Evaluation---
    for eval_category in eval_pools:
        print((f"Evaluation for alignment -- '{eval_category}'"))

        # --- Setup Paths ---
        image_dir = (
            args.image_path / eval_category
        )
        caption_path = project_root / "captions" / f"{language_code}" / f"{eval_category}.jsonl"
        
        output_base_dir = score_root_dir / "alignment"

        # --- Initialize API ---
        api_config = ApiConfig(model=args.api_model, api_key=args.api_key, base_url=args.base_url)
        client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)

        # --- Load Data ---
        messages_pool = get_message_templates()
        messages_template = messages_pool.get("alignment")[eval_category]
        if not messages_template:
            print(f"Error: No message template found for category '{eval_category}'")
            return

        category_descriptions = load_category_descriptions(caption_path)
        if not category_descriptions:
            print(f"Error: No descriptions found in '{caption_path}'")
            return

        # --- Main Processing Loop ---
        for i in range(args.num_images):
            img_path = image_dir / f"{i}.png"

            save_path = (
                output_base_dir
                / eval_category
                / f"{i}.jsonl"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.exists():
                print(f"Result already exists at '{save_path}'. Skipping.")
                continue

            if not img_path.exists():
                print(f"Image does not exist at '{img_path}'. Skipping.")
                continue

            start_time = time.time()

            try:
                image = Image.open(img_path)
                base64_image = encode_image_to_base64(image)
            except Exception as e:
                print(f"Failed to open or encode image '{img_path}': {e}")
                continue

            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }

            current_messages = deepcopy(messages_template)
            current_messages[1]["content"].append(image_message)
            current_messages[1]["content"][0]["text"] = current_messages[1]["content"][0][
                "text"
            ].format(text_prompt=category_descriptions[i])

            output_text = get_model_response(
                client,
                current_messages,
                model=api_config.model,
                max_tokens=4096,
                temperature=0.0,
                top_p=1.0,
            )

            if not output_text:
                print("Skipping saving due to empty API response.")
                continue

            print("Output: ", output_text)
            result_data = clean_and_parse_json(output_text)

            if not result_data:
                print("Skipping saving due to JSON parsing failure.")
                continue

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
            print(
                f"Result saved to '{save_path}'. Time taken: {time.time() - start_time:.2f}s"
            )

        print((f"------------------- Evaluation for alignment -- '{eval_category}' -- End -------------------"))

    print((f"------------------- Evaluation for alignment End -------------------"))

    # --- Start Aesthetic Evaluation---
    for eval_category in eval_pools:
        print((f"Evaluation for aesthetic -- '{eval_category}'"))

        # --- Setup Paths ---
        image_dir = (
            args.image_path / eval_category
        )
        caption_path = project_root / "captions" / f"{language_code}" / f"{eval_category}.jsonl"
        
        output_base_dir = score_root_dir / "aesthetic"

        # --- Initialize API ---
        api_config = ApiConfig(model=args.api_model, api_key=args.api_key, base_url=args.base_url)
        client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)

        # --- Load Data ---
        messages_pool = get_message_templates()
        messages_template = messages_pool.get("aesthetic")
        if not messages_template:
            print(f"Error: No message template found for category '{eval_category}'")
            return

        category_descriptions = load_category_descriptions(caption_path)
        if not category_descriptions:
            print(f"Error: No descriptions found in '{caption_path}'")
            return

        # --- Main Processing Loop ---
        for i in range(args.num_images):
            img_path = image_dir / f"{i}.png"

            save_path = (
                output_base_dir
                / eval_category
                / f"{i}.jsonl"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.exists():
                print(f"Result already exists at '{save_path}'. Skipping.")
                continue

            if not img_path.exists():
                print(f"Image does not exist at '{img_path}'. Skipping.")
                continue

            start_time = time.time()

            try:
                image = Image.open(img_path)
                base64_image = encode_image_to_base64(image)
            except Exception as e:
                print(f"Failed to open or encode image '{img_path}': {e}")
                continue

            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }

            current_messages = deepcopy(messages_template)
            current_messages[1]["content"].append(image_message)
            current_messages[1]["content"][0]["text"] = current_messages[1]["content"][0][
                "text"
            ].format(text_prompt=category_descriptions[i])

            output_text = get_model_response(
                client,
                current_messages,
                model=api_config.model,
                max_tokens=4096,
                temperature=0.0,
                top_p=1.0,
            )

            if not output_text:
                print("Skipping saving due to empty API response.")
                continue

            print("Output: ", output_text)
            result_data = clean_and_parse_json(output_text)

            if not result_data:
                print("Skipping saving due to JSON parsing failure.")
                continue

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
            print(
                f"Result saved to '{save_path}'. Time taken: {time.time() - start_time:.2f}s"
            )
        
        print((f"------------------- Evaluation for aesthetic -- '{eval_category}' -- End -------------------"))

    print((f"------------------- Evaluation for aesthetic End -------------------"))

    # --- Summarize and Analyze Results ---
    print((f"Start organizing final results"))
    summarize_results(score_root_dir, eval_pools, args.num_images)

if __name__ == "__main__":
    main()
    