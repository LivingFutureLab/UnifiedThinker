import base64
import os
import json
import argparse
import tempfile
import math
from PIL import Image
import requests
from termcolor import colored
import io
import re
import logging
import random
import oss2
import pandas as pd
import hashlib
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


api_key=""
MODEL= "gemini-2.5-flash-image-preview"


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_prompt_file', type=str, default="/data/oss_bucket_0/jianchong.zq/tmp/test_metadata_mix_zss.jsonl")
parser.add_argument('--outdir', type=str, default="/data/oss_bucket_0/jianchong.zq/datas/gemini_generate_images/")
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--pdb_debug', action='store_true')
args = parser.parse_args()


input_datas = []
with open(args.input_prompt_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        if line == "": continue
        tmp = json.loads(line)
        if isinstance(tmp, dict):
            input_datas.append(tmp)
        elif isinstance(tmp, (tuple, list)):
            input_datas.extend(tmp)
        else:
            raise ValueError()
print(f"Find {len(input_datas)} datas")

for i, d in enumerate(input_datas):
    d['index'] = i
    
output_dir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.input_prompt_file))[0])
os.makedirs(output_dir, exist_ok=True)


def decode_base64_to_pil(base64_str: str) -> Image.Image | None:
    match = re.search(r'base64,([A-Za-z0-9+/=]+)', base64_str)
    b64_data = match.group(1) if match else base64_str
    try:
        image_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"图像解码失败: {e}")
        return None

# 定义哪些异常需要重试。通常是网络相关或API临时性错误。
# openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError 都是值得重试的。
RETRYABLE_EXCEPTIONS = (
    Exception  # 为了简单，这里仍然捕获所有异常，但在生产环境中建议更具体
)
      
# 使用 tenacity 装饰器实现健壮的重试逻辑
@retry(
    stop=stop_after_attempt(5),  # 最多重试5次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 等待时间：2s, 4s, 8s, 10s, 10s
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logging.warning(
        f"Retrying API call... Attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep:.2f} seconds."
    )
)
def call_vision_api(prompt):
    """
    调用多模态视觉生成 API（支持多个 base64 图像输入）
    返回模型输出文本或图像链接
    # 
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # 构建消息内容
    content = [{"type": "text", "text": prompt}]

    payload = {
        "messages": [{"role": "user", "content": content}],
        "model": MODEL,
        "stream": False,
        "modalities": ["TEXT"]
    }
    
    try:
        resp = requests.post(
            "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions", 
            headers=headers, 
            data=json.dumps(payload), 
            timeout=300
        )
        resp.raise_for_status()
        result = resp.json()

        msg_content = result['choices'][0]['message']['content']
        finish_reason = result['choices'][0]['finish_reason']

        if not msg_content or finish_reason == 'content_filter':
            print(f"生成被阻止: {finish_reason}")
            return None
        return msg_content

    except requests.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"解析响应失败: {e}")
    return None
    
          
def process_data(data):
    """
    处理单个图片的完整逻辑：加载、调用API、解析结果。
    这个函数将被并行调用。
    """    
    prompt = data['prompt']
    output_file = os.path.join(output_dir, f"{data['index']}.png")
    
    if os.path.isfile(output_file):
        logging.info(f"{output_file} already processed ...")
    try:
        api_response = call_vision_api(prompt)
        if api_response:
            try:
                img_url_in_resp = api_response[0]['image_url']['url']  # 注意：取决于实际返回结构
                generated_img = decode_base64_to_pil(img_url_in_resp)
                if generated_img:
                    generated_img.save(output_file)
                    logging.info(f"Saved generated image to {output_file}")
            except Exception as e:
                logging.error(f"Failed to save generated image for prompt '{prompt}'. Error: {e}")
            
    except Exception as e:
        # 捕捉所有可能的异常（包括API调用失败、图片加载失败等）
        logging.error(f"Failed to process prompt '{prompt}'. Error: {e}", exc_info=True)
        

if __name__ == "__main__":
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        
    success_count = 0
                
    if args.pdb_debug:
        process_data(input_datas[0])
        
    else:
        # 使用线程池执行并行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 提交所有任务
            future_to_data = {executor.submit(process_data, d): d for d in input_datas}
            
            # 使用tqdm显示进度条，并处理已完成的任务
            for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(input_datas), desc="Processing Images"):
                data_item = future_to_data[future]
                try:
                    result = future.result()
                    success_count += 1
                except Exception as e:
                    logging.error(f"An error occurred while processing {data_item.get('prompt')}", exc_info=True)
                    
        logging.info(f"All tasks completed. {success_count} out of {len(input_datas)} images processed successfully.")
    