#coding=utf-8
#: 改为 qwenvl 进行 judge, 看看是不是能够作为 reward model
#支持多卡

import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable
import time
import re
import pandas as pd # 确保导入pandas，因为代码中使用了pd
from utils import * # 假设 load, dump 等函数在此文件中
from tqdm import tqdm
import multiprocessing as mp
from termcolor import colored

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info


subtask_dic = {
    "Temp": [
        "Life Progression",
        "Material Progression",
        "Environmental Cycles",
        "Societal Transformation",
    ],
    "Causal": [
        "Structural Deformation",
        "State Transition",
        "Chemical and Biological Transformation",
        "Physics Manifestation",
    ],
    "Spa": [
        "Component Assembly",
        "Object Arrangement",
        "Viewpoint Generation",
        "Structural Inference",
        "Layout Reasoning",
    ],
    "Logic": ["Pattern Prediction", "Mathematical Derivation", "Puzzle Solving"],
}

# ==============================================================================
# 新函数: 使用 Qwen-VL 模型进行生成
# ==============================================================================
def qwen_vl_generate(model, vlm_processor, inputs, temperature=0, max_tokens=4096, **kwargs):
    """
    使用本地加载的 Qwen-VL 模型生成文本。
    Args:
        model: 加载好的 Qwen-VL 模型对象。
        vlm_processor: 加载好的分词器对象。
        inputs (list): 包含文本和图片信息的列表，格式为
                       [{'type': 'text', 'value': '...'}, {'type': 'image', 'value': 'path/to/img.jpg'}]
        temperature (float): 生成温度。
        max_tokens (int): 最大生成 token 数。
    Returns:
        tuple: (ret_code, answer, response_object)
               ret_code: 0表示成功，-1表示失败。
               answer: 模型生成的文本回答。
               response_object: 在此为 None。
    """
    # 兼容原有的 temperature 设置，如果 temperature 为 0，则使用贪心解码
    gen_kwargs = {"max_new_tokens": max_tokens}
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})

    try:
        prompt_parts = []
        for item in inputs:
            if item['type'] == 'text':
                prompt_parts.append({"type": "text", "text": item['value']})
            elif item['type'] == 'image':
                prompt_parts.append({"type": "image", "image": Image.open(item['value']).convert("RGB")})

        messages = [{"role": "user", "content": prompt_parts}]
        
        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(model.device)
        generated_ids = model.generate(**model_inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        return 0, answer, None

    except Exception as e:
        print(f"❌ Error during Qwen-VL generation: {e}")
        print(f"Problematic inputs: {inputs}")
        return -1, f"Failed to obtain answer due to error: {e}", None

def find_image(output_dir, index):
    for suffix in ['png', 'jpg', 'jpeg']:
        img_path = osp.join(output_dir, f"{index}.{suffix}")
        if osp.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Cannot find output images {index} in {output_dir}!!!")

# ==============================================================================
# 修改: eval_vanilla 函数签名和内部调用
# ==============================================================================
def eval_vanilla(item, input_dir, output_dir, model, vlm_processor, **kwargs):
    instruct = item['instruction']
    index = item['index']
    category = item['category']
    output_dir = osp.join(output_dir, f'images/{category}')
    img2 = find_image(output_dir, index)
    judge_exist = item.get('judge', None)
    judge_rea_require_img = False

    
    if category in ['temporal_reasoning', 'causal_reasoning']:
        img1 = osp.join(input_dir, item['image'])
        reference = item['reference']
        if "reference_img" in item and not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            prompt_rea = prompt_reasoning_w_input.format(instruct=instruct, reference=reference)
        else:
            prompt_rea = prompt_reasoning.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_consist.format(instruct=instruct)
        prompt_qua = prompt_generation

    elif category == 'spatial_reasoning':
        img1 = osp.join(input_dir, item['image'])
        if "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img = True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_rea = prompt_spatial_ref_img.format(instruct=instruct)
        elif not pd.isna(item['reasoning_img']):
            judge_rea_require_img = True
            reference = item['reference']
            prompt_rea = prompt_spatial_ref_w_input.format(instruct=instruct, reference=reference)
        else:
            reference = item['reference']
            prompt_rea = prompt_spatial_ref.format(instruct=instruct, reference=reference)

        prompt_cons = prompt_spatial_cons.format(instruct=instruct)
        prompt_qua = prompt_spatial_qual

    elif category == 'logical_reasoning':
        if "reference_txt" in item and not pd.isna(item['reference_txt']):
            img1 = osp.join(input_dir, item['image'])
            reference = item['reference_txt']
            prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=reference)
            prompt_rea = prompt_logical_txt.format(instruct=instruct, reference=reference)
        elif "reference_img" in item and not pd.isna(item['reference_img']):
            judge_rea_require_img=True
            img1 = osp.join(input_dir, item['reference_img'])
            prompt_cons = prompt_logical_cons.format(instruct=instruct)
            if 'reasoning_wo_ins' in item:
                prompt_rea = prompt_logical_img_wo_q
            else:
                prompt_rea = prompt_logical_img.format(instruct=instruct)

    if 'consistency_free' in item and not pd.isna(item['consistency_free']):
        consist_judge = None
        print('Consistency Judgement not required. Ignore.')
    else:
        if judge_exist and 'judge1' in judge_exist:
            consist_judge = judge_exist['judge1']
        else:
            message = [
                {'type': 'text', 'value': prompt_cons},
                {'type': 'image', 'value': img1},
                {'type': 'image', 'value': img2}
            ]
            print(f"Generating for judge1 (consistency) on index {index}...")
            # 调用新的生成函数
            ret_code, consist_judge, response = qwen_vl_generate(model, vlm_processor, message, **kwargs)

    if judge_exist and 'judge2' in judge_exist:
        answer2 = judge_exist['judge2']
    else:
        if judge_rea_require_img:
            message2 = [
                {'type': 'text', 'value': prompt_rea}, 
                {'type': 'image','value': img1},
                {'type': 'image','value': img2}
                ]
        else:
            message2 = [
                {'type': 'text', 'value': prompt_rea},
                {'type': 'image', 'value': img2}
            ]
        print(f"Generating for judge2 (reasoning) on index {index}...")
        # 调用新的生成函数
        ret_code2, answer2, response2 = qwen_vl_generate(model, vlm_processor, message2, **kwargs)

    if category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        if judge_exist and 'judge3' in judge_exist:
            answer3 = judge_exist['judge3']
        else:
            message3 = [
                {'type': 'text', 'value': prompt_qua},
                {'type': 'image', 'value': img2}
            ]
            print(f"Generating for judge3 (quality) on index {index}...")
            # 调用新的生成函数
            ret_code3, answer3, response3 = qwen_vl_generate(model, vlm_processor, message3, **kwargs)

        return dict(judge1=consist_judge, judge2=answer2, judge3=answer3)
    else:
        return dict(judge1=consist_judge, judge2=answer2)

# ==============================================================================
# 新增: 多卡并行工作进程函数
# ==============================================================================
def worker(worker_id, 
           #device_id, 
           gpus_per_model,
           model_path, input_queue, output_queue, input_dir, output_dir):
    """
    每个工作进程执行的函数。
    它会加载模型到指定的GPU，然后从输入队列获取任务并处理，最后将结果放入输出队列。
    """
    # # 关键步骤: 隔离每个进程的可见GPU
    # gpu_start_index = worker_id * gpus_per_model
    # gpu_end_index = gpu_start_index + gpus_per_model
    # visible_devices = ",".join(map(str, range(gpu_start_index, gpu_end_index)))
    # os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
    # # 在子进程中重新导入 torch，确保它能正确识别被隔离的设备
    # import torch
    # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration
    
    # print(f"[Worker {worker_id}] Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # print(f"[Worker {worker_id} on GPU {visible_devices}] Loading model...")
    # print(f"Worker {worker_id} 进程可见的 GPU 数量: {torch.cuda.device_count()}") # 这应该会打印 2
    # print(f"Worker {worker_id} 当前设备: {torch.cuda.current_device()}")      # 这应该会打印 0
    
    if 'qwen3' in model_path.lower():
        model_class = Qwen3VLMoeForConditionalGeneration
    else:
        model_class = Qwen2_5_VLForConditionalGeneration
        
    try:        
        processor = AutoProcessor.from_pretrained(model_path)
        if gpus_per_model == 1:
            device_id = worker_id
            model = model_class.from_pretrained(
                model_path, torch_dtype="auto", device_map=f"cuda:{device_id}"
            ).eval()
        else:
            from accelerate import infer_auto_device_map, init_empty_weights
            
            gpu_start_index = worker_id * gpus_per_model
            gpu_end_index = gpu_start_index + gpus_per_model
    
            with init_empty_weights():
                empty_model = model_class.from_pretrained(model_path, torch_dtype="auto")

            # 2. 使用 accelerate 的辅助函数来自动生成一个分布在 GPU 0 和 1 上的映射字典
            # 你可以为每个设备指定内存限制，以指导 accelerate 如何分配
            # 注意：这里的 0 和 1 是物理 GPU 索引
            device_map_dict = infer_auto_device_map(
                empty_model,
                max_memory={e: "80GiB" for e in range(gpu_start_index, gpu_end_index)},
                no_split_module_classes=empty_model._no_split_modules
            )
            
            model = model_class.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device_map_dict # <--- 将 "cuda:0,1" 修改为一个字典
            )
        
    except Exception as e:
        print(f"[Worker {worker_id} ❌ Failed to load model: {e}")
        # 发送一个错误信号给主进程
        output_queue.put(f"ERROR_WORKER_{worker_id}")
        return

    # 循环处理任务
    while True:
        # 从队列中获取一个任务项
        item = input_queue.get()

        # 如果收到 None，表示任务结束
        if item is None:
            break

        try:
            # 调用评估函数
            result_dict = eval_vanilla(
                item=item,
                input_dir=input_dir,
                output_dir=output_dir,
                model=model,
                vlm_processor=processor
            )
            # 将 (任务索引, 结果) 元组放入输出队列
            output_queue.put((item['index'], result_dict))
        except Exception as e:
            print(f"[Worker {worker_id}] ❌ Error processing item {item.get('index', 'N/A')}: {e}")
            output_queue.put((item.get('index', 'N/A'), None)) # 即使失败也返回，避免主进程卡死
                      
def extract(answer):
    if not answer: return None
    # (其余代码保持不变)
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    else:
        return None

def calculate_score(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        if 'consistency_free' in row and row['consistency_free']:
            score = 0.2 * row['VisualPlausibility'] + 0.8 * row['Reasoning']
        else:
            score = 0.3 * row['ApprConsistency'] + 0.5 * row['Reasoning'] + 0.2 * row['VisualPlausibility']
        
    elif row['category'] == 'logical_reasoning':
        score = 0.3 * row['ApprConsistency'] + 0.7 * row['Reasoning']
    if row['Reasoning'] == 1:
        score = score * 0.5
        score = 1 if score < 1 else score
    return score

def calculate_completion(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        return (
            1
            if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 and row['VisualPlausibility'] == 5
            else 0
        )
    elif row['category']=='logical_reasoning':
        return (
            1 if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 else 0
        )

# : 过滤出不成功的cases
def filter_result(result):
    failed_indexs = []
    for index in result:
        judge = result[index]
        if judge['judge1'] is None:
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score2 or not score3:
                score=None
            else:
                score = [None]+score2+score3
        elif 'judge3' not in judge:
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        elif 'judge2' not in judge:
            score = [extract(judge['judge1'])[1], extract(judge['judge1'])[0]]
        else:
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score1 or not score2 or not score3:
                score=None
            else:
                score = score1+score2+score3
        
        if score is None:
            failed_indexs.append(index)
    
    print(colored(f"Total {len(result)} sampels, of which {len(failed_indexs)} fialed samples", "green", attrs=["bold"]))
    for index in failed_indexs:
        result.pop(index)
    print(colored(f"After filtering, left {len(result)} sampels", "green", attrs=["bold"]))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Json Path')
    parser.add_argument('--output', type=str, required=True, help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--input', type=str, default='data', help='Input Image Dir')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default="judage_by_qwen_vl", help='Model Name')
    
    parser.add_argument('--pretrained_model', type=str)

    # 修改: nproc 含义变为使用的 GPU 数量
    #parser.add_argument('--nproc', type=int, default=1, help='Number of GPUs to use for parallel processing.')
    parser.add_argument('--gpus_per_model', type=int, default=1, help=" >1 for large model")

    args = parser.parse_args()
    #import pdb; pdb.set_trace()
    
    # 安全地设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # 如果已经设置，则忽略
    
    # 确定要使用的进程数
    num_gpus = torch.cuda.device_count()
    if args.gpus_per_model > 1:
        assert num_gpus % args.gpus_per_model == 0 and num_gpus >= args.gpus_per_model
    nproc = num_gpus // args.gpus_per_model

    # # ==============================================================================
    # # 新增: 加载 Qwen-VL 模型和分词器
    # # ==============================================================================
    model_name = os.path.basename(args.pretrained_model.rstrip('/'))
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge.xlsx"
        score_file = f"{args.output}/{model_name}_judge.csv"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_judge.csv"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    data = json.load(open(args.data))
    data = pd.DataFrame(data)

    result = {}
    if osp.exists(tmp_file):
        result = load(tmp_file)
    result = filter_result(result)

    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    # # ==============================================================================
    # # 修改: 将 model 和 vlm_processor 传入评测任务
    # # ==============================================================================    
    if not items:
        print("All items have already been processed. Skipping evaluation.")
    # 条件: nproc > 1 且有足够多的GPU时，启动并行处理
    elif nproc > 1:
        print(f"🚀 Starting parallel evaluation for {len(items)} items using {nproc} works...")

        input_queue = mp.Queue()
        output_queue = mp.Queue()

        # 将所有待处理任务放入输入队列
        for item in items:
            input_queue.put(item)
        # 为每个工作进程添加一个结束信号
        for _ in range(nproc):
            input_queue.put(None)

        # 创建并启动工作进程
        processes = []
        for i in range(nproc):
            p = mp.Process(
                target=worker,
                args=(i, args.gpus_per_model, args.pretrained_model, input_queue, output_queue, args.input, args.output)
            )
            p.start()
            processes.append(p)

        # 从输出队列收集结果，并显示进度条
        for _ in tqdm(range(len(items)), desc="Processing items"):
            res = output_queue.get()
            # 检查是否有 worker 启动失败
            if isinstance(res, str) and res.startswith("ERROR_WORKER"):
                print(f"FATAL: {res} failed to initialize. Aborting.")
                # 终止所有子进程
                for p in processes:
                    p.terminate()
                return # 提前退出

            index, single_result = res
            if single_result is not None:
                result[index] = single_result
            
            # 实时保存，防止中断
            if len(result) % 10 == 0: # 每处理10个保存一次
                dump(result, tmp_file)
        
        # 等待所有工作进程结束
        for p in processes:
            p.join()

        dump(result, tmp_file) # 最后再保存一次
    
    # 否则，使用原始的单卡顺序执行逻辑
    else:
        print(f"🐢 Starting sequential evaluation for {len(items)} items on a single GPU...")
        
        print("Loading Qwen2.5-VL-32B-Chat model... This may take a while.")
        model_id = args.pretrained_model
        
        vlm_processor = AutoProcessor.from_pretrained(model_id)
        
        if 'qwen3' in model_path.lower():
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            model_class = Qwen2_5_VLForConditionalGeneration
        
        # 使用 device_map="auto" 让 transformers 自动管理单机多卡（模型并行）或单卡
        model = model_class.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        ).eval()
        print("Model loaded successfully.")

        for item in tqdm(items, desc="Evaluating items"):
            key = item['index']
            try:
                single_result = eval_vanilla(
                    item=item,
                    input_dir=args.input,
                    output_dir=args.output,
                    model=model,
                    vlm_processor=vlm_processor
                )
                result[key] = single_result
                dump(result, tmp_file) # 每处理完一个就保存
            except Exception as e:
                print(f"\n--- ❌ Error processing item with index {key}: {e} ---")
                continue
    
    print("Evaluation finished. Calculating scores...")

    # ==============================================================================
    # 后续的计分和报告生成部分保持不变
    # ==============================================================================
    print("Evaluation finished. Calculating scores...")
    judges = [result[i] for i in data['index']]

    scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []

    for judge in judges:
        if judge is None: # 添加一个检查，以防生成失败
            scores.append(None)
            judge_combine.append(None)
            judge_cons.append(None)
            judge_reas.append(None)
            judge_qua.append(None)
            continue
            
        if judge.get('judge1') is None:
            judge_combine.append(
                'REASONING\n\n'
                + str(judge.get('judge2', ''))
                + '\n\nQUALITY\n\n'
                + str(judge.get('judge3', ''))
            )
            judge_cons.append(None)
            judge_reas.append(judge.get('judge2'))
            judge_qua.append(judge.get('judge3'))

            score2 = extract(judge.get('judge2'))
            score3 = extract(judge.get('judge3'))
            if not score2 or not score3:
                score=None
            else:
                score = [None]+score2+score3
        elif 'judge3' not in judge:
            judge_combine.append(
                'CONSISTENCY\n\n'
                + str(judge.get('judge1', ''))
                + '\n\nREASONING\n\n'
                + str(judge.get('judge2', ''))
            )
            judge_cons.append(judge.get('judge1'))
            judge_reas.append(judge.get('judge2'))
            judge_qua.append(None)

            score1 = extract(judge.get('judge1'))
            score2 = extract(judge.get('judge2'))
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        elif 'judge2' not in judge:
            judge_combine.append(judge.get('judge1'))
            score = [extract(judge.get('judge1'))[1], extract(judge.get('judge1'))[0]]
        else:
            try:
                judge_combine.append(
                    'CONSISTENCY\n\n'
                    + str(judge['judge1'])
                    + '\n\nREASONING\n\n'
                    + str(judge['judge2'])
                    + '\n\nQUALITY\n\n'
                    + str(judge['judge3'])
                )
                judge_cons.append(judge.get('judge1'))
                judge_reas.append(judge.get('judge2'))
                judge_qua.append(judge.get('judge3'))
            except Exception as e:
                print(e)
                breakpoint()
            score1 = extract(judge.get('judge1'))
            score2 = extract(judge.get('judge2'))
            score3 = extract(judge.get('judge3'))
            if not score1 or not score2 or not score3:
                score=None
            else:
                score = score1+score2+score3
        scores.append(score)

    reasoning = []
    img_consist = []
    gen_quality = []
    match_log = []

    for score in scores:
        if score:
            match_log.append('succeed')
            if len(score)==3:
                img_consist.append(score[0])
                reasoning.append(score[1])
                gen_quality.append(score[2])

            elif len(score)==2:
                reasoning.append(4 * min(score[1], 1) + 1)
                img_consist.append(4 * min(score[0], 1) + 1)
                gen_quality.append(None)
        else:
            img_consist.append(None)
            reasoning.append(None)
            gen_quality.append(None)
            match_log.append('failed')
    
    data['Reasoning'] = reasoning
    data['ApprConsistency'] = img_consist
    data['VisualPlausibility'] = gen_quality
    data['match_log'] = match_log
    data['judge_cons'] = judge_cons
    data['judge_reas'] = judge_reas
    data['judge_qua'] = judge_qua
    
    print(colored(f"match_log: {len(match_log)} samples; 其中 {sum([e == 'failed' for e in match_log])} failed samples.", "green", attrs=["bold"]))

    # 填充NaN值，以便计算分数
    for col in ['ApprConsistency', 'Reasoning', 'VisualPlausibility']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        # data[col] = data[col].fillna(0) # or some other default value
    
    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    dump(data.to_dict('records'), judge_res.replace('.xlsx', '.json')) # 使用dump保存为json或pkl
    try:
        data.to_excel(judge_res, index=False) # 保存为Excel
    except Exception as e:
        print("error of {}".format(str(e)))
    
    df_causal = data[data['category'] == 'causal_reasoning']
    df_temporal = data[data['category'] == 'temporal_reasoning']
    df_spatial = data[data['category'] == 'spatial_reasoning']
    df_logical = data[data['category'] == 'logical_reasoning']

    score_final = data['score'].mean()
    completion_rate = data['complete'].mean()
    
    temporal_final, temporal_comp_rate = df_temporal['score'].mean(), df_temporal['complete'].mean()
    causal_final, causal_comp_rate = df_causal['score'].mean(), df_causal['complete'].mean()
    spatial_final, spatial_comp_rate = df_spatial['score'].mean(), df_spatial['complete'].mean()
    logical_final, logical_comp_rate = df_logical['score'].mean(), df_logical['complete'].mean()

    reasoning_average = data['Reasoning'].mean()
    img_consist_average = data['ApprConsistency'].mean()
    generation_quality = data['VisualPlausibility'].mean()

    temp_rea_avg, temp_cons_avg, temp_qua_avg = df_temporal['Reasoning'].mean(), df_temporal['ApprConsistency'].mean(), df_temporal['VisualPlausibility'].mean()
    cau_rea_avg, cau_cons_avg, cau_qua_avg = df_causal['Reasoning'].mean(), df_causal['ApprConsistency'].mean(), df_causal['VisualPlausibility'].mean()
    spa_rea_avg, spa_cons_avg, spa_qua_avg = df_spatial['Reasoning'].mean(), df_spatial['ApprConsistency'].mean(), df_spatial['VisualPlausibility'].mean()
    logic_rea_avg, logic_cons_avg, logic_qua_avg = df_logical['Reasoning'].mean(), df_logical['ApprConsistency'].mean(), df_logical['VisualPlausibility'].mean()

    def trans_to_percent(s):
        if pd.isna(s): return None
        return 25*(s-1)
    
    average_scores_by_subtask = data.groupby('subtask')['score'].mean()
    average_acc_by_subtask = data.groupby('subtask')['complete'].mean()

    average_scores_dict = average_scores_by_subtask.to_dict()
    average_acc_dict = average_acc_by_subtask.to_dict()
    
    subtask_results = {}
    for k, v in average_scores_dict.items():
        subtask_results[k] = [v, trans_to_percent(v), average_acc_dict.get(k)]
    
    sorted_subtask_results = {}
    for main_task_prefix, subtasks in subtask_dic.items():
        for subtask in subtasks:
            if subtask in subtask_results:
                new_key = f"{main_task_prefix}-{subtask}"
                sorted_subtask_results[new_key] = subtask_results[subtask]

    final_score = dict(
        Overall=[score_final, trans_to_percent(score_final), completion_rate],
        Temporal=[temporal_final, trans_to_percent(temporal_final), temporal_comp_rate],
        Causal=[causal_final, trans_to_percent(causal_final), causal_comp_rate],
        Spatial=[spatial_final, trans_to_percent(spatial_final), spatial_comp_rate],
        Logical=[logical_final, trans_to_percent(logical_final), logical_comp_rate],
        Overall_Reasoning=[reasoning_average, trans_to_percent(reasoning_average), None],
        Overall_ApprConsistency=[img_consist_average, trans_to_percent(img_consist_average), None],
        Overall_VisualPlausibility_total=[generation_quality, trans_to_percent(generation_quality), None],
        Temporal_Reasoning = [temp_rea_avg, trans_to_percent(temp_rea_avg), None],
        Temporal_Consistency = [temp_cons_avg, trans_to_percent(temp_cons_avg), None],
        Temporal_Quality = [temp_qua_avg, trans_to_percent(temp_qua_avg), None],
        Causal_Reasoning = [cau_rea_avg, trans_to_percent(cau_rea_avg), None],
        Causal_Consistency = [cau_cons_avg, trans_to_percent(cau_cons_avg), None],
        Causal_Quality = [cau_qua_avg, trans_to_percent(cau_qua_avg), None],
        Spatial_Reasoning = [spa_rea_avg, trans_to_percent(spa_rea_avg), None],
        Spatial_Consistency = [spa_cons_avg, trans_to_percent(spa_cons_avg), None],
        Spatial_Quality = [spa_qua_avg, trans_to_percent(spa_qua_avg), None],
        Logical_Reasoning = [logic_rea_avg, trans_to_percent(logic_rea_avg), None],
        Logical_Consistency = [logic_cons_avg, trans_to_percent(logic_cons_avg), None],
        **sorted_subtask_results
    )
    
    print("-" * 100)
    print(final_score)
    print("-" * 100)

    df = pd.DataFrame(final_score, index=["Score-Origin", "Score-Percentage", "Accuracy"]).T
    df.reset_index(inplace=True)
    df.columns = ["-", "Score-Origin", "Score-Percentage", "Accuracy"]
    df.to_csv(score_file, index=False)
    print(f"Scores saved to {score_file}")
    print(f"Detailed judge results saved to {judge_res}")


if __name__ == '__main__':
    main()