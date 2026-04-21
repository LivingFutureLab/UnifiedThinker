#coding=utf-8
#jianchong.zq: 本地调试

import os
import sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import ray
import torch
from PIL import Image

# 导入 Hydra 和 dacite 相关库
from hydra import initialize, compose
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig

from roll.utils.import_utils import safe_import_class
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.datasets.collator import DataCollatorWithPaddingForMM
from rlvr_image_think.rlvr_image_think_config import RLVRImageThinkConfig
from rlvr_image_think.rlvr_image_think_pipeline import get_extra_data_provider

# ==================== 用户需要修改的配置 ====================

# 要加载的主配置文件名 (不带 .yaml 后缀)
CONFIG_NAME = "rlvr_image_think_debug"  

# 请指定您想要调试的 reward model 对应的 domain/key
REWARD_DOMAIN_TO_DEBUG = "reason_edit" 

# ==========================================================

def create_mock_image(size=(224, 224), color='red'):
    """创建一个假的 PIL.Image.Image 对象用于测试"""
    return Image.new('RGB', size, color)


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    # --- 启动 Ray 本地模式 ---
    print("Initializing Ray in local mode...")
    ray.init(local_mode=True)

    # 使用 with 上下文管理器来初始化 Hydra
    # 这可以防止全局状态污染
    with initialize(config_path="./", job_name="debug_app"):
        # 组合配置
        cfg = compose(config_name=CONFIG_NAME)        
        print(OmegaConf.to_yaml(cfg, resolve=True))

        # 将 OmegaConf 对象转换为 RLVRImageThinkConfig 实例
        config_container = OmegaConf.to_container(cfg, resolve=True)
        pipeline_config = from_dict(
            data_class=RLVRImageThinkConfig, 
            data=config_container, 
            config=DaciteConfig(strict=False) # 使用 non-strict 模式以忽略多余的配置项
        )

    print("RLVRImageThinkConfig object created successfully.")

    # --- 4. 模拟 Worker 环境 ---
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # --- 5. 准备并实例化 Reward Worker ---
    print(f"Targeting reward worker for domain: '{REWARD_DOMAIN_TO_DEBUG}'")

    if REWARD_DOMAIN_TO_DEBUG not in pipeline_config.rewards:
        raise KeyError(
            f"Domain '{REWARD_DOMAIN_TO_DEBUG}' not found in the `rewards` section of your config. "
            f"Available domains: {list(pipeline_config.rewards.keys())}"
        )
    reward_worker_config = pipeline_config.rewards[REWARD_DOMAIN_TO_DEBUG]
    RewardWorkerClass = safe_import_class(reward_worker_config.worker_cls)

    print(f"Instantiating worker class: {RewardWorkerClass.__name__}")
    
    RemoteRewardWorker = ray.remote(RewardWorkerClass)
    reward_actor = RemoteRewardWorker.remote(worker_config=reward_worker_config)
    
    # --- 调用 initialize 方法 ---
    print("Calling worker's initialize() method... (Set a breakpoint inside to debug model loading)")
    ray.get(reward_actor.initialize.remote(pipeline_config=pipeline_config))
    print("Initialization complete.")

    # --- 准备模拟输入数据 (Mock Data) ---
    print("Preparing mock data for reward computation...")
    
    processor = default_processor_provider(pipeline_config.actor_train.model_args)      # <class 'transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor'>
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    mock_raw_data = [
        {
            "prompt": "将天空变成日落时的橙色",
            "images": [create_mock_image(color='blue')],
            "responses": "一张图片，天空呈现出绚丽的日落橙色，云彩被染上了金色和粉色的边缘。",
            "domain": REWARD_DOMAIN_TO_DEBUG, 
            "reward_model": REWARD_DOMAIN_TO_DEBUG,
        },
    ]

    collator = DataCollatorWithPaddingForMM(
        processor=processor,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=pipeline_config.prompt_length,
        extra_unpadded_keys=["domain", "reward_model"],
        extra_data_provider=get_extra_data_provider(
                            pipeline_config.actor_train.model_args.model_name_or_path, processor=processor
                        ), 
        prompt_key="prompt",
        answer_key="responses",
        image_key="images",
        image_flag_key=None
    )
    
    # batch_dict = collator(mock_raw_data)
    # mock_data_proto = DataProto(batch=batch_dict, meta_info={"is_training": True})
    
    collated_output = collator(mock_raw_data)
    # ==================== 核心修复点 (适配 DataProto) ====================
    import numpy as np

    # 1. 将 collator 的输出（BatchEncoding）转换为标准字典。
    # 2. 将 'extra_unpadded_keys' (如 domain) 从 list 转换为 numpy array，
    #    以满足 DataProto.from_single_dict 的要求。
    processed_dict = {}
    for key, value in collated_output.items():
        if isinstance(value, list):
            # DataProto 的 non_tensor_batch 期望 np.ndarray，使用 dtype=object 可以存储字符串等
            processed_dict[key] = np.array(value, dtype=object)
        else:
            # 其他类型（如 torch.Tensor）保持不变
            processed_dict[key] = value

    # 3. 使用 DataProto 提供的工厂方法来正确创建实例。
    #    这个方法会自动将字典中的 Tensors 和 non-tensors 分离开。
    mock_data_proto = DataProto.from_single_dict(
        data=processed_dict,
        meta_info={"is_training": True}
    )
    # =====================================================================
    
    # --- 8. 调用核心计算方法 (compute_rewards / forward) ---
    REWARD_METHOD_NAME = "compute_rewards"
    print(f"▶️  Calling worker's `{REWARD_METHOD_NAME}` method... (Set a breakpoint inside to debug reward logic)")

    if not hasattr(reward_actor, REWARD_METHOD_NAME):
        raise AttributeError(f"The worker does not have a method named '{REWARD_METHOD_NAME}'. "
                             "Please check your worker's implementation and update the REWARD_METHOD_NAME variable.")

    reward_method = getattr(reward_actor, REWARD_METHOD_NAME)
    
    result_proto_ref = reward_method.remote(data=mock_data_proto)
    result_proto = ray.get(result_proto_ref)

    print("✅ Reward computation complete.")
    
    if "scores" in result_proto.batch:
        scores = result_proto.batch["scores"]
        print("\n--- Reward Scores ---")
        print(scores.tolist())
        print("---------------------\n")
    else:
        print("Warning: 'scores' not found in the result. The result batch contains:")
        print(result_proto.batch.keys())

    # --- 9. 关闭 Ray ---
    print(" shutting down Ray.")
    ray.shutdown()
    
    