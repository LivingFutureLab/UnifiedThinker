#coding=utf-8
#: xxx

import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import types
from PIL import Image
import glob
from termcolor import colored
import oss2
import json
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist

from peft import LoraConfig
from transformers import AutoProcessor
from safetensors import safe_open

from src.pipe.pipeline_qwen_image_edit_think import QwenImageEditThinkPipeline
from src.data.odps_t2i_edit_data import preprocess_image
from src.model.utils import download_model_weight
from inference.qwen_edit_think.demo_qwen_edit_think import vlm_prompt_thinking


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

def get_risebench_datas():
    meta_file = "/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask_only_temp_cuasal.json"
    datas = json.load(open(meta_file))
    
    outdatas = []
    for d in datas:
        prompt = d["instruction"]
        image_pre_edit = os.path.join("/data/oss_bucket_0//benchmarks/RISEBench/data/", d["image"])
        
        outdatas.append({
            "task": "edit",
            "image_pre_edit": image_pre_edit,
            "text": prompt,
            "output_image_suffix": "images/{}/{}.png".format(d["category"], d["index"])
        })
    return outdatas

def get_gedit_datas(edit_raw_image=1, language="cn"):
    meta_file = "/data/oss_bucket_0/mllm_dataset/public_datasets/image_edit/GEdit-Bench/gedit_1k.json"
    datas = json.load(open(meta_file))
    image_dir = os.path.join(os.path.dirname(meta_file), "images")
    outdatas = []
    for d in datas:
        if d["instruction_language"] != language:
            continue
        
        if edit_raw_image > 0:
            image_pre_edit = os.path.join(image_dir, "{}_raw.png".format(d['key']))
        else:
            image_pre_edit = os.path.join(image_dir, "{}.png".format(d['key']))
        text = d["instruction"]

        outdatas.append({
            "task": "edit",
            "image_pre_edit": image_pre_edit,
            "text": text,
            "output_image_suffix_pre_edit": "fullset/{}/{}/{}_SRCIMG.png".format(d["task_type"], d["instruction_language"], d["key"]),
            "output_image_suffix": "fullset/{}/{}/{}.png".format(d["task_type"], d["instruction_language"], d["key"]),
        })
    return outdatas
   
def upload_file(bucket, local_file_path, oss_file_path):
    try:
        bucket.put_object_from_file(oss_file_path, local_file_path)
    except Exception as e:
        print(f'Failed to upload {local_file_path} to {oss_file_path}: {e}')
            
import oss2
from concurrent.futures import ThreadPoolExecutor
def oss2_upload_folder(local_folder, oss_folder, args):
    auth = oss2.Auth(args.oss_access_id, args.oss_access_key)
    bucket = oss2.Bucket(auth, args.oss_endpoint, args.bucket_name)
    
    def list_files(local_folder, oss_folder):
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder)
                oss_file_path = os.path.join(oss_folder, relative_path).replace('\\', '/')
                yield local_file_path, oss_file_path
                
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for local_file_path, oss_file_path in list_files(local_folder, oss_folder):
            future = executor.submit(upload_file, bucket, local_file_path, oss_file_path)
            futures.append(future)
        for future in futures:
            future.result()
   
def load_model(args):    
    if args.pretrained_model.startswith("model."):
        # download from mos
        args.pretrained_model = download_model_weight(args.pretrained_model)
    else:
        assert os.path.exists(args.pretrained_model), "{} not exist.".format(args.pretrained_model)
    dist.barrier()
    
    pipe = QwenImageEditThinkPipeline.from_pretrained(
        args.pretrained_model, 
        torch_dtype=torch.bfloat16)

    if args.text_encoder_lora:
        text_encoder_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        pipe.text_encoder.language_model.add_adapter(text_encoder_lora_config)
    
    if args.transformer_lora:
        transformer_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipe.transformer.add_adapter(transformer_lora_config)
        
    if args.train_ckpt_file is not None:
        if args.train_ckpt_file.startswith("model."):
            args.train_ckpt_file = download_model_weight(args.train_ckpt_file)
        else:
            assert os.path.exists(args.train_ckpt_file)
        dist.barrier()
            
        # # 切片保存
        index_path = os.path.join(args.train_ckpt_file, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        weight_map = index_data["weight_map"]
        
        shard_files = set(weight_map.values())
        tensors_by_shard = {shard: [] for shard in shard_files}
        for tensor_name, shard_file in weight_map.items():
            tensors_by_shard[shard_file].append(tensor_name)
        
        final_state_dict = {}
        # 使用tqdm来显示进度条
        for shard_file, tensor_names in tqdm(tensors_by_shard.items(), desc="Loading shards"):
            shard_path = os.path.join(args.train_ckpt_file, shard_file)
            
            # 使用 safe_open，它不会立即加载所有张量到内存，而是创建一个映射
            # 这对于内存非常友好
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_name in tensor_names:
                    # 从分片文件中获取单个张量
                    tensor = f.get_tensor(tensor_name)
                    # 放入我们最终的 state_dict 中
                    final_state_dict[tensor_name] = tensor
        
        state_dict_transformer = {k.replace("transformer.", ""): v for k, v in final_state_dict.items() if k.startswith("transformer.")}
        state_dict_textencoder = {k.replace("text_encoder.", ""): v for k, v in final_state_dict.items() if k.startswith("text_encoder.")}
        assert len(final_state_dict) == len(state_dict_transformer) + len(state_dict_textencoder)
        
        if len(state_dict_transformer) > 0:
            missing_keys, unexpected_keys = pipe.transformer.load_state_dict(state_dict_transformer, strict=False)
            print("Load {} params for transformer, with {} unexpected_keys".format(len(state_dict_transformer), len(unexpected_keys)))
        if len(state_dict_textencoder) > 0:
            missing_keys, unexpected_keys = pipe.text_encoder.load_state_dict(state_dict_textencoder, strict=False)
            print("Load {} params for text_encoder, with {} unexpected_keys".format(len(state_dict_textencoder), len(unexpected_keys)))
    
    return pipe

def get_caption_language(prompt):
    ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
        # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
        # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
    ]
    for char in prompt:
        if any(start <= char <= end for start, end in ranges):
            return 'zh'
    return 'en'  
      
      
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_model', type=str, help="pretrained model path")
    parser.add_argument('--train_ckpt_file', type=str, help="finetuned ckpt path")
    
    parser.add_argument('--vlm_processor_path', type=str, default="/data/oss_bucket_0//pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct/")
    
    parser.add_argument('--text_encoder_lora', type=int, default=0)
    parser.add_argument('--transformer_lora', type=int, default=0)
    
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=64)
    
    parser.add_argument('--think', type=int, default=0)
        
    parser.add_argument('--dataset', type=str, default="risebench", choices=["risebench", "gedit"])
    #parser.add_argument('--language', type=str, default="cn", choices=["en", "cn"])
    #parser.add_argument('--gedit_raw_image', type=int, default=1)
    
    parser.add_argument('--upload2oss', action='store_true')     
    parser.add_argument("--oss_access_id", type=str)
    parser.add_argument("--oss_access_key", type=str)
    parser.add_argument("--oss_endpoint", type=str)
    parser.add_argument("--bucket_name", type=str)
    
    parser.add_argument('--pdb_debug', action='store_true')
    args = parser.parse_args()
    
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        
    if args.pretrained_model[-1] == "/":
        args.pretrained_model = args.pretrained_model[:-1]
        print(f"pretrained_model: {args.pretrained_model}")
    if args.train_ckpt_file is not None and args.train_ckpt_file[-1] == "/":
        args.train_ckpt_file = args.train_ckpt_file[:-1]
        print(f"train_ckpt_file: {args.train_ckpt_file}")
    
    if not args.pdb_debug:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(get_local_rank())

    if args.dataset == "risebench":
        datas = get_risebench_datas()
    elif args.dataset == "gedit":
        datas = get_gedit_datas(language="cn", edit_raw_image=1)
    else:
        raise ValueError()
    
    print("Total {} samples".format(len(datas)))
    if not args.pdb_debug:
        print("rank: {}, world_size: {}".format(get_global_rank(), get_world_size()))
        datas = datas[get_global_rank()::get_world_size()]
        print("rank-{} has {} samples".format(get_global_rank(), len(datas)))
        device = torch.device("cuda:{}".format(get_local_rank()))
    else:
        device = "cuda"
    
    # prepare outdir
    if args.train_ckpt_file is None:
        if os.path.exists(args.pretrained_model):
            outdir = os.path.join(args.pretrained_model, args.dataset)
        else: # mos
            outdir = "{}/{}".format(args.pretrained_model.split("version=")[-1], args.dataset)
    else:
        if os.path.exists(args.train_ckpt_file):
            outdir = os.path.join(args.train_ckpt_file, args.dataset)
        else: # mos
            outdir = "{}/{}".format(args.train_ckpt_file.split("version=")[-1], args.dataset)

    if args.think > 0:
        outdir = outdir + "_think"
        
    if args.train_ckpt_file is None:
        # qwen-image-edit 使用固定 pixel area;
        fix_ref_img_pixel_area = True
    else:
        # ours 是否应该和训练保持一致: 只对 > target_area 的图片进行下采样，不会对 < target_area 的图片进行上采样;
        fix_ref_img_pixel_area = False
    print(colored(f"fix_ref_img_pixel_area set to {fix_ref_img_pixel_area}", "green", attrs=["bold"]))

    pipe = load_model(args)
    pipe = pipe.to(device)
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_processor_path)
        
    auth = oss2.Auth(args.oss_access_id, args.oss_access_key)
    bucket = oss2.Bucket(auth, args.oss_endpoint, args.bucket_name)
            
    for data in tqdm(datas):
        torch.cuda.empty_cache()
        
        prompt = data["text"]
        
        if args.dataset in ["risebench", "gedit"]:
            image_pre_edit = data["image_pre_edit"]
            image_pre_edit = Image.open(image_pre_edit).convert("RGB")
            image_pre_edit = preprocess_image(image_pre_edit, max_area=1024*1024, adjust_ar=False)
            ref_imgs = [image_pre_edit]
            tgt_width, tgt_height = ref_imgs[0].size
        else:
            raise ValueError("not supported dataset: {}".format(args.dataset))

        if args.think > 0:
            prompt_cot = vlm_prompt_thinking(ref_imgs, prompt, vlm_processor, pipe, max_new_tokens=256)
            print(colored(f"prompt_cot: {prompt_cot}", "green", attrs=["bold"]))
        else:
            prompt_cot = None
            
        outfile = os.path.join(outdir, data["output_image_suffix"])
        inputs = {
            "image": ref_imgs,
            "prompt": prompt,
            "prompt_cot": prompt_cot,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "fix_ref_img_pixel_area": fix_ref_img_pixel_area
        }
        with torch.inference_mode():
            output = pipe(**inputs, height=tgt_height, width=tgt_width)
            image = output.images[0]
        
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        image.save(outfile)
        print("write to {}".format(outfile))
        if args.upload2oss:
            oss_file = os.path.join("/tbstar_image_eval_results", outfile)
            upload_file(bucket, outfile, oss_file)
        
        if args.dataset == "gedit":
            outfile = os.path.join(outdir, data["output_image_suffix_pre_edit"])
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            os.system("cp {} {}".format(data["image_pre_edit"], outfile))
            print("write to {}".format(outfile))
            if args.upload2oss:
                oss_file = os.path.join("/tbstar_image_eval_results", outfile)
                upload_file(bucket, outfile, oss_file)
                                    
    dist.barrier()
    print("rank-{} done".format(get_global_rank()))
    dist.destroy_process_group()
    