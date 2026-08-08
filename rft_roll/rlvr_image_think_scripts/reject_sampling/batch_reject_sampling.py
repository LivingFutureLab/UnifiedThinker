#coding=utf-8
# Input data: (pre_edit_image, edit_prompt, gt_cot)
#   Step-1: 目标模型 (e.g., qwen-edit) 首先根据 (pre_edit_image, edit_prompt) 进行 cot的 rollout;
#   Step-2: qwen-edit 模型使用 cot 预测 post-images;
#   Step-3: RM 根据 (pre_edit_image, edit_prompt, gt_cot, post_image) 进行打分；


import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import tempfile
import textwrap
import random
import datetime
import math
from tqdm import tqdm
from PIL import Image
import json 
from copy import deepcopy
from termcolor import colored
import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

from rlvr_image_think.image_edit_think_pipe.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from rlvr_image_think.image_edit_think_pipe.util import preprocess_image
from rlvr_image_think.utils import download_model_mos
from rlvr_image_think.image_edit_reward_worker import extract_scores_from_text, calculate_final_score, qwen_vl_generate


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "-1"))

def get_global_rank() -> int:
    return int(os.getenv("RANK", "0"))

def get_world_size(group=None) -> int:
    if group is not None:
        return dist.get_world_size(group)
    return int(os.getenv("WORLD_SIZE", "1"))

def is_local_first_process() -> bool:
    mdl_local_rank = int(os.getenv("LOCAL_PROCESS_RANK", "-1"))
    if mdl_local_rank != -1:
        return mdl_local_rank == 0
    else:
        return get_local_rank() in (-1, 0)

def is_world_first_process() -> bool:
    return get_local_rank() == -1 or get_global_rank() in (-1, 0)


def vlm_prompt_thinking(ref_images, edit_prompt, think_model, max_new_tokens=256):
    content = []
    for im in ref_images:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": edit_prompt})
        
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "你是一位专业的视觉推理专家和图像编辑顾问。你的核心任务是：\n1.  **深入分析**: 接收用户提供的原始图片和编辑指令（edit prompt）。\n2.  **逻辑推理**: 结合图片内容（如物体材质、所处环境、当前状态）和生活常识、物理规律或特定艺术风格等，对编辑指令进行深度推理，预测出指令执行后最可能产生的视觉结果。\n3.  **精准描述**: 将推理出的编辑后图像样貌，用一段精炼、客观、富有画面感的文字描述出来。\n\n要求：\n- 直接返回最终的图像内容描述。\n- 描述内容控制在200字以内。\n- 禁止包含任何解释、分析过程或多余的客套话。"}]
        },
        {
            "role": "user",
            "content": content
        }
    ]
        
    # Preparation for inference
    inputs = think_model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(think_model.device)

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = think_model.generate(**inputs, 
                                max_new_tokens=max_new_tokens,
                                do_sample=True,          # 启用采样，这是增加多样性的关键
                                temperature=0.8,         # 温度参数，值越高，随机性越强，但可能降低连贯性。0.7-0.9 是常用范围。
                                top_p=0.9)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    thinked_prompt = think_model.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(thinked_prompt) == 1, "batch is not 1"
    return thinked_prompt[0]

def infer_qwen_edit(ref_imgs, prompt, pipe):
    ref_imgs = [Image.open(im).convert("RGB") if isinstance(im, str) else im for im in ref_imgs]
    ref_imgs  = [preprocess_image(im, max_area=1024*1024, adjust_ar=False) for im in ref_imgs]
            
    width, height = ref_imgs[0].size
    print(f"width, height: {width} x {height}")
    print(colored(f"prompt: {prompt}", "green", attrs=["bold"]))
    
    inputs = {
        "image": ref_imgs,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "fix_ref_img_pixel_area": True
    }

    with torch.inference_mode():
        output = pipe(**inputs, height=height, width=width)
        output_image = output.images[0]
    return output_image

def infer_reward_score(pre_edit_image, edit_prompt, edit_prompt_cot, post_edit_image, vlm_model):
    # 和 rft_roll/rlvr_image_think/image_edit_reward_worker.py 保持一致            
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
    _, judge1, _ = qwen_vl_generate(vlm_model, vlm_model.processor, message, temperature=0, max_tokens=1024)
    
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
    _, judge2, _ = qwen_vl_generate(vlm_model, vlm_model.processor, message2, temperature=0, max_tokens=1024)

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
    _, judge3, _ = qwen_vl_generate(vlm_model, vlm_model.processor, message3, temperature=0, max_tokens=1024)
    
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
    return final_score


import oss2
oss_bucket = oss2.Bucket(oss2.Auth(oss_access_id, oss_access_key), oss_endpoint, bucket_name)

def load_image_from_oss(oss_path) -> str:
    object_key = oss_path.replace(f"oss://{bucket_name}/", "")
    object_key = object_key.replace("/data/oss_bucket_0/", "").replace("/data/oss_bucket_1/", "")
    suffix = os.path.splitext(object_key)[1]
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_f:
        oss_bucket.get_object_to_file(object_key, tmp_f.name)
        image = Image.open(tmp_f.name).convert("RGB")
        return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str )
    parser.add_argument('--thinker_model', type=str, default="/tmp/jianchong.zq/checkpoints/Qwen2.5-VL-7B-Instruct/")
    parser.add_argument('--edit_model', type=str, default="/tmp/jianchong.zq/checkpoints/Qwen-Image-Edit-2509/")
    parser.add_argument('--score_model', type=str, default="/tmp/jianchong.zq/checkpoints/Qwen3-VL-8B-Instruct/")
    parser.add_argument('-n', '--num_rollouts', type=int, default=16)
    
    parser.add_argument('--pdb_debug', action='store_true')
    args = parser.parse_args()
    
    if not args.pdb_debug:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(get_local_rank())
    else:
        import pdb; pdb.set_trace()
        
    datas = []
    with open(args.data_file, 'r') as f:
        for line in f.readlines():
            datas.append(json.loads(line.strip()))
    
    print("Total {} samples".format(len(datas)))
    if not args.pdb_debug:
        print("rank: {}, world_size: {}".format(get_global_rank(), get_world_size()))
        datas = datas[get_global_rank()::get_world_size()]
        print("rank-{} has {} samples".format(get_global_rank(), len(datas)))
        device = torch.device("cuda:{}".format(get_local_rank()))
    else:
        device = torch.device("cuda:0")
    
    # 1.1 加载 thinker 模型
    thinker_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.thinker_model, torch_dtype=torch.bfloat16
    )
    thinker_model = thinker_model.to("cpu") # 加载到CPU
    thinker_model.processor = AutoProcessor.from_pretrained(args.thinker_model)
    
    # 1.2 加载 image-edit 模型
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.edit_model, 
        torch_dtype=torch.bfloat16)
    pipe = pipe.to("cpu")   # 加载到CPU
    
    # 1.3 加载 RM 模型
    if "qwen2.5-vl" in args.score_model.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "qwen3-vl" in args.score_model.lower():
        if "qwen3-vl-8b" in args.score_model.lower():
            model_class = Qwen3VLForConditionalGeneration
        elif "qwen3-vl-30b-a3b" in args.score_model.lower():
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            raise ValueError(f"not supported model: {args.score_model}")
    else:
        raise ValueError()
        
    args.score_model = download_model_mos(args.score_model)   
    score_model = model_class.from_pretrained(args.score_model, torch_dtype=torch.bfloat16)
    score_model = score_model.to("cpu") # 加载到CPU
    score_model.processor = AutoProcessor.from_pretrained(
        args.score_model,
        trust_remote_code=True
    )    
    
    out_datas = []
    for data in tqdm(datas):         
        ######################## step 1: prompt thinking #########################
        print("\nOffloading other models, loading Thinker model to GPU...")
        pipe.to('cpu')
        score_model.to('cpu')
        thinker_model.to(device)
        torch.cuda.empty_cache()
        
        ref_imgs_oss_path = data["ref_imgs"]
        ref_images_pil = [load_image_from_oss(p) for p in ref_imgs_oss_path]
        
        edit_prompt = data["edit_prompt"]
        edit_prompt_cot = data["edit_prompt_cot"]

        rollout_cots = []
        for _ in range(args.num_rollouts):
            try:
                torch.cuda.empty_cache()
                cot = vlm_prompt_thinking(
                    ref_images_pil, edit_prompt, thinker_model, max_new_tokens=256
                )
                rollout_cots.append(cot)
            except Exception as e:
                print("error of {}".format(str(e)))
                rollout_cots.append(None)
        print("Step1: Generated {} cots".format(len(rollout_cots)))
        for cot in rollout_cots:
            if cot is not None:
                print(cot)
        
        
        ##################### step 2: post edit image generation ###################
        print("Offloading Thinker model, loading Edit model to GPU...")
        thinker_model.to('cpu')
        score_model.to('cpu')
        pipe.to(device)
        torch.cuda.empty_cache()
                
        edited_images = []
        for cot in rollout_cots:
            try:
                assert cot is not None, "Generated cot is None"
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # 使用预测的 cot 替代 edit_prompt
                    edited_image = infer_qwen_edit(ref_images_pil, cot, pipe)
                    edited_images.append(edited_image)
            except Exception as e:
                print("error of {}".format(str(e)))
                edited_images.append(None)
        
        
        ##################### step 3: generate reward score ###################
        print("Offloading Edit model, loading Score model to GPU...")
        thinker_model.to('cpu')
        pipe.to('cpu')
        score_model.to(device)
        torch.cuda.empty_cache()
        
        reward_scores = []
        for post_edit_image in edited_images:
            if post_edit_image is None:
                continue
            #注意: 在计算 reward score 时，使用 generated post_edit_image, 但是使用 gemini合成的高质量edit_prompt_cot 作为打分参考
            try:
                torch.cuda.empty_cache()
                score = infer_reward_score(ref_images_pil, edit_prompt, edit_prompt_cot, post_edit_image, score_model)
                reward_scores.append(score)
            except Exception as e:
                print("error of {}".format(str(e)))
        
        print("收集到有效rewad scores: {}".format(len(reward_scores)))
        data["reward_scores_for_rs"] = reward_scores
        out_datas.append(data)
        
        if len(out_datas) % 20 == 0:
            out_file = os.path.join(os.path.splitext(args.data_file)[0] + "_reject_sampling", "rs_results_rank_{}.jsonl".format(get_global_rank()))
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                for d in out_datas:
                    f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
            print(f"write to {out_file}")
            
    dist.barrier()
    
    if is_world_first_process():
        out_datas = []
        for i in range(get_world_size()):
            out_file = os.path.join(os.path.splitext(args.data_file)[0] + "_reject_sampling", "rs_results_rank_{}.jsonl".format(i))
            with open(out_file, 'r') as f:
                for line in f.readlines():
                    out_datas.append(json.loads(line.strip()))
            
        out_file = os.path.join(os.path.splitext(args.data_file)[0] + "_reject_sampling", "rs_results.jsonl")
        with open(out_file, 'w') as f:
            for d in out_datas:
                f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
        print(f"write to {out_file}")
    
    dist.destroy_process_group()