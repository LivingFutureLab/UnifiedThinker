import os
import re
from PIL import Image
from io import BytesIO
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union
from openlm_hub import repo_download
import time
import datasets
from datasets import Dataset, Image as HFImage
import oss2
import json
from tqdm import tqdm
import concurrent.futures
from functools import partial

from roll.utils.logging import get_logger
from roll.utils.checkpoint_manager import download_model, model_path_cache, file_lock_context

from rlvr_image_think.image_edit_think_pipe.util import preprocess_image


logger = get_logger()


@contextlib.contextmanager
def temporary_env_var(key, value):
    """一个安全的上下文管理器，用于临时设置环境变量。"""
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            # 如果原来不存在，就删除它
            del os.environ[key]
        else:
            # 否则，恢复原值
            os.environ[key] = original_value

@model_path_cache
def download_model_mos(model_name_or_path: str, local_dir: Optional[str] = None):
    # 从mos载模型
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    
    if local_dir is None:
        local_dir = f"./ckpt_temp_{model_name_or_path.split('/')[-1]}"
    
    #assert model_name_or_path.startswith("model."), f"check {model_name_or_path} is a vliad mos model"
    if not model_name_or_path.startswith("model."):
        return download_model(model_name_or_path, local_dir)    # download from HF or modelscope
        
    success_flag_path = os.path.join(local_dir, ".download_successful")
    
    with file_lock_context(model_name_or_path):
        # 进入锁之后，再次检查是否已经被其他进程下载完成, 检查的是“成功标记”而不是目录是否存在
        if os.path.exists(success_flag_path):
            print(f"Model already downloaded and verified in {local_dir}. Skipping.")
            return local_dir
        
        try:
            with temporary_env_var("USER_ID", "147878"):
                print(f"Mos model loading to {local_dir}, From {model_name_or_path}")
                repo_download(repo_id=model_name_or_path, local_dir=local_dir)
            
            with open(success_flag_path, 'w') as f:
                f.write('success')
            print(f"Successfully downloaded and created success flag in {local_dir}")

        except Exception as e:
            print(f"Failed to download model {model_name_or_path}. Error: {e}")
            raise  # 重新抛出异常，让上层知道失败了
        
        return local_dir
    
def pre_download_and_update_dataset(
    dataset: Dataset,
    image_column: str,
    cache_path: str,
    oss_config: dict,
    num_proc: int = 32
) -> Dataset:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    
    success_file = os.path.join(cache_path, ".download_success")
    local_image_root = os.path.join(cache_path, "downloaded_images")

    # Rank 0 负责下载的逻辑
    if rank == 0 and not os.path.exists(success_file):
        # 使用 cache_path 作为锁的唯一标识符。
        # 您的函数会对其进行哈希，所以这很安全。
        logger.info("Rank 0: Attempting to acquire download lock...")
        with file_lock_context(cache_path):
            # 双重检查：在获取锁后，再次检查是否已完成
            if os.path.exists(success_file):
                logger.info("Rank 0: Acquired lock, but found success file. Skipping download.")
            else:
                logger.info("Rank 0: Acquired lock, starting download process...")
                
                auth = oss2.Auth(oss_config['access_id'], oss_config['access_key'])
                bucket = oss2.Bucket(auth, oss_config['endpoint'], oss_config['bucket_name'])
                os.makedirs(local_image_root, exist_ok=True)

                def download_worker(oss_path):
                    if not oss_path: return None
                    try:
                        relative_path = oss_path.replace(f"oss://{oss_config['bucket_name']}/", "").replace("/data/oss_bucket_0/", "").replace("/data/oss_bucket_1/", "")
                        local_path = os.path.join(local_image_root, relative_path)
                        if not os.path.exists(local_path):
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            bucket.get_object_to_file(relative_path, local_path)
                        return
                    except Exception as e:
                        logger.error(f"Rank 0 Download Error: {oss_path}, {e}")
                        return

                all_oss_paths = set(p for paths in dataset[image_column] if paths for p in (paths if isinstance(paths, list) else [paths]) if p)
                unique_paths = list(all_oss_paths)
                
                if unique_paths:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                        list(tqdm(executor.map(download_worker, unique_paths), total=len(unique_paths), desc="Rank 0 Downloading"))
                
                # 下载完成后，创建成功标志
                with open(success_file, "w") as f:
                    f.write("done")
                logger.info("Rank 0: Download complete. Success file created. Lock released.")

    # 所有进程（包括 rank 0）都等待，确保下载（如果需要）已完成
    if not os.path.exists(success_file):
        logger.info(f"Rank {rank}: Waiting for rank 0 to finish downloading...")
        while not os.path.exists(success_file):
            time.sleep(10)
    logger.info(f"Rank {rank}: Success file found. Proceeding.")

    # --- 路径映射和数据集更新逻辑 ---
    path_map_file = os.path.join(cache_path, "image_path_map.json")
    path_map = {}
    
    if rank == 0 and not os.path.exists(path_map_file):
        logger.info("Rank 0: Creating path map file...")
        all_oss_paths = set(p for paths in dataset[image_column] if paths for p in (paths if isinstance(paths, list) else [paths]) if p)
        for oss_path in all_oss_paths:
            relative_path = oss_path.replace(f"oss://{oss_config['bucket_name']}/", "").replace("/data/oss_bucket_0/", "").replace("/data/oss_bucket_1/", "")
            path_map[oss_path] = os.path.join(local_image_root, relative_path)
        with open(path_map_file, 'w') as f:
            json.dump(path_map, f)
        logger.info("Rank 0: Path map file created.")
    
    while not os.path.exists(path_map_file):
        time.sleep(5)
        
    with open(path_map_file, 'r') as f:
        path_map = json.load(f)

    logger.info(f"Rank {rank}: Updating dataset with local paths...")
    def update_path_function(example, path_map):
        oss_paths = example[image_column]
        if not oss_paths: return example
        if isinstance(oss_paths, str):
            example[image_column] = path_map.get(oss_paths)
        elif isinstance(oss_paths, list):
            example[image_column] = [path_map.get(p) for p in oss_paths if p in path_map]
        return example
        
    updated_dataset = dataset.map(
        partial(update_path_function, path_map=path_map),
        num_proc=num_proc,
        desc=f"Rank {rank} updating paths"
    )
    return updated_dataset

def pre_download_and_load_images(
    dataset: Dataset,
    image_column: str,
    oss_config: dict,
    num_proc: int = 32,
    max_pixels: int = 1024 * 1024
) -> Dataset:
    """
    并行地从OSS下载图片到内存，直接加载为 PIL.Image 对象，
    并用这些对象更新数据集。
    Args:
        dataset (Dataset): 原始数据集。
        image_column (str): 包含图片OSS路径的列名。
        oss_config (dict): OSS连接配置。
        num_proc (int): dataset.map 使用的进程数。

    Returns:
        Dataset: 一个新的数据集，其中 image_column 已被替换为 PIL.Image 对象列表。
    """
    if image_column not in dataset.column_names:
        logger.warning(
            f"Image column '{image_column}' not found in dataset. "
            "Skipping image loading and returning original dataset."
        )
        return dataset
    
    # 只需要 rank 0 来执行下载到内存的操作，其他进程等待结果
    # 但由于最终的 map 需要所有进程都参与，我们在这里不做 rank 判断，
    # 而是在 map 的 function 内部处理（虽然 map 本身在这里只跑一次）
    logger.info("Starting image pre-loading from OSS into memory...")

    # --- 1. 收集所有唯一的图片路径 ---
    all_oss_paths = set(
        p for paths in dataset[image_column] if paths 
        for p in (paths if isinstance(paths, list) else [paths]) if p
    )
    unique_paths = list(all_oss_paths)
    logger.info(f"Found {len(unique_paths)} unique images to load into memory.")

    # --- 并行下载所有图片到内存中 ---
    # 这个map存储 oss_path -> PIL.Image
    image_memory_map = {}
    failed_downloads = []

    if unique_paths:
        auth = oss2.Auth(oss_config['access_id'], oss_config['access_key'])
        bucket = oss2.Bucket(auth, oss_config['endpoint'], oss_config['bucket_name'])
        
        def download_to_memory_worker(oss_path):
            """下载到内存，返回 (oss_path, PIL.Image)"""
            if not oss_path: return oss_path, None
            try:
                # 统一处理路径
                relative_path = oss_path.replace(f"oss://{oss_config['bucket_name']}/", "")
                path_prefix_pattern = re.compile(r"/data/oss_bucket_\d+/")
                relative_path = path_prefix_pattern.sub("", relative_path)
                
                # 从OSS获取对象到内存
                obj = bucket.get_object(relative_path)
                image_bytes = obj.read()
                
                # 从字节流加载为PIL Image对象
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                if max_pixels is not None:
                    pil_image = preprocess_image(pil_image, max_area=max_pixels, adjust_ar=False)
                
                return oss_path, pil_image
            except Exception as e:
                logger.error(f"Download-to-memory Error: {oss_path}, {e}")
                return oss_path, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            future_to_path = {executor.submit(download_to_memory_worker, path): path for path in unique_paths}
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(unique_paths), desc="Loading images into memory"):
                original_oss_path, pil_image = future.result()
                if pil_image:
                    image_memory_map[original_oss_path] = pil_image
                else:
                    failed_downloads.append(original_oss_path)

    # --- 检查失败并中止 ---
    if len(failed_downloads) > 100:
        logger.error(f"Total {len(failed_downloads)} images failed to load into memory.")
        logger.error("First 10 failed paths: " + str(failed_downloads[:10]))
        raise RuntimeError("Image loading failed, stopping preprocessing to prevent data corruption.")
    
    logger.info("All images successfully loaded into memory.")

    # --- 4. 使用 map 一次性替换数据集中的路径为 PIL 对象 ---
    logger.info("Updating dataset with in-memory PIL.Image objects...")
    
    def replace_path_with_image_object(example, image_map):
        """将单个样本中的路径替换为内存中的 PIL.Image 对象"""
        oss_paths = example[image_column]
        if not oss_paths:
            # 确保即使没有路径，列的类型也是正确的（一个空列表）
            example[image_column] = []
            return example
        
        if isinstance(oss_paths, str):
            # 如果原先是单个字符串，替换为包含单个Image对象的列表
            img_obj = image_map.get(oss_paths)
            example[image_column] = [img_obj] if img_obj else []
        elif isinstance(oss_paths, list):
            # 替换为Image对象的列表，过滤掉None（下载失败或路径不存在的）
            example[image_column] = [image_map.get(p) for p in oss_paths if p in image_map]
            
        return example

    # 定义新的数据集特征，告诉datasets库'images'列现在是Image对象列表
    features = dataset.features.copy()
    features[image_column] = HFImage()
    if isinstance(dataset.features[image_column], datasets.Sequence):
         features[image_column] = datasets.Sequence(HFImage())

    updated_dataset = dataset.map(
        partial(replace_path_with_image_object, image_map=image_memory_map),
        num_proc=num_proc,
        features=features, # 应用新的特征定义
        desc="Replacing paths with PIL objects"
    )
    return updated_dataset
