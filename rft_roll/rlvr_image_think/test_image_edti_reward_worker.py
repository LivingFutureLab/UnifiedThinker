import os, sys
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict
from tensordict import TensorDict

import hydra
from omegaconf import OmegaConf, DictConfig

from roll.configs.worker_config import WorkerConfig, ModelArguments, StrategyArguments
from roll.distributed.scheduler.protocol import DataProto
from rlvr_image_think.image_edit_reward_worker import ImageEditRewardWorker
from rlvr_image_think.rlvr_image_think_config import ImageEditRewardConfig



# def create_mock_config() -> ImageEditRewardConfig:
#     """
#     创建一个模拟的 ImageEditRewardConfig, 用于本地测试。
#     """
#     # 使用 EasyDict 可以方便地创建类似 YAML 解析后的嵌套字典结构
#     mock_config = EasyDict()

#     mock_config.name = "image_edit_reward_worker_test"
#     mock_config.worker_cls = "rlvr_image_think.image_edit_reward_worker.ImageEditRewardWorker"
#     mock_config.device_mapping = '[0]' # 指定用于测试的 GPU ID
#     mock_config.num_gpus_per_worker = 1

#     # 定义两个模型集群的配置
#     mock_config.clusters = [
#         # Cluster 0: Image Edit Model
#         {
#             "name": "image-edit-model",
#             "model_args": {
#                 "model_name_or_path": "/tmp/jianchong.zq/checkpoints/Qwen-Image-Edit-2509/",
#             },
#             "strategy_args": {
#                 "strategy_name": "hf_infer", # 使用HuggingFace标准推理策略
#                 "generation_args": {},
#             },
#         },
#         # Cluster 1: VLM Judge Model
#         {
#             "name": "vlm-judge-model",
#             "model_args": {
#                 "model_name_or_path": "/tmp/jianchong.zq/checkpoints/Qwen2.5-VL-7B-Instruct/",
#                 "model_max_length": 4096,
#             },
#             "strategy_args": {
#                 "strategy_name": "hf_infer", # 使用HuggingFace标准推理策略
#                 "generation_args": {},
#             },
#         },
#     ]

#     # 将 EasyDict 转换为 WorkerConfig 对象
#     worker_config = ImageEditRewardConfig(**mock_config)
#     return worker_config

def create_config_from_yaml(config_path: str, config_name: str) -> ImageEditRewardConfig:
    """
    从指定的 YAML 文件中加载配置并创建 ImageEditRewardConfig 实例。
    """
    print(f"--- 正在从 '{os.path.join(config_path, config_name)}.yaml' 加载配置 ---")
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg: DictConfig = hydra.compose(config_name=config_name)

    # 2. 提取 'rewards.reason_edit' 部分的配置
    if not ('rewards' in cfg and 'reason_edit' in cfg.rewards):
        raise KeyError("在 YAML 文件中未找到 'rewards.reason_edit' 配置节。")
    
    reward_worker_yaml_config = cfg.rewards.reason_edit
    print("成功提取 'rewards.reason_edit' 配置。")

    worker_config = ImageEditRewardConfig(**reward_worker_yaml_config)
    print("WorkerConfig 实例创建成功。")

    return worker_config

def create_mock_data() -> DataProto:
    """
    创建一个模拟的 DataProto 对象，包含测试所需的输入数据。
    此版本已根据您提供的 DataProto 定义进行修正。
    """
    # 1. 创建虚拟输入图片和文本
    dummy_image = Image.new('RGB', (512, 512), 'red')
    edit_prompt = "make the image blue"
    edit_prompt_cot = "The user wants to change the color of the entire image from red to blue. The final image should be a solid blue square."

    # 2. 构建 non_tensor_batch，并遵循 DataProto 的要求
    non_tensor_batch = {
        "pre_edit_images": np.array([[dummy_image]], dtype=object),
        "edit_prompt": np.array([edit_prompt], dtype=object),
        "edit_prompt_cot": np.array([edit_prompt_cot], dtype=object),
    }

    # 3. 创建 DataProto 对象
    mock_data = DataProto(
        # 修正点 1：将 'tensor_batch' 重命名为 'batch'
        batch=TensorDict({}, batch_size=[1]),
        non_tensor_batch=non_tensor_batch,
        meta_info={"global_step": 1, "is_offload_states": True}
    )
    return mock_data


def main():
    import pdb; pdb.set_trace()
    
    try:
        print("--- 从 YAML 创建配置并创建模拟数据 ---")       
        # 使用新函数从 YAML 加载配置
        worker_config = create_config_from_yaml("../rlvr_image_think_scripts", "rlvr_image_think_8gpu")
        mock_data_proto = create_mock_data()
        
        # 从 YAML 加载 system_envs
        with hydra.initialize(config_path="../rlvr_image_think_scripts", version_base=None):
            cfg = hydra.compose(config_name="rlvr_image_think_8gpu")
            pipeline_config = EasyDict(OmegaConf.to_container(cfg, resolve=True))

        print(f"\n--- 实例化 {worker_config.worker_cls} ---")
        device_id = worker_config.device_mapping[0]
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"将在设备 {device} 上运行测试 (基于 YAML 配置: {worker_config.device_mapping})")
        
        worker = ImageEditRewardWorker(worker_config=worker_config)
        worker.rank_info.rank = 0
        worker.rank_info.world_size = 1 # 本地测试，world_size 始终为 1
        worker.rank_info.device = device
        
        worker.logger.setLevel("INFO")

        print("\n--- 调用 worker.initialize() ---")
        worker.initialize(pipeline_config=pipeline_config)
        print("Worker 初始化成功，模型已加载。")

        print("\n--- 调用 worker.compute_rewards() ---")
        final_scores = worker.compute_rewards(data=mock_data_proto)

        print("\n--- 测试完成，查看结果 ---")
        print(f"计算得到的最终奖励分数: {final_scores}")
        
        assert isinstance(final_scores, list), "结果应该是一个列表"
        assert len(final_scores) == 1, "批处理大小为1，应返回一个分数"
        assert isinstance(final_scores[0], float), "分数应该是浮点数"
        print("\n✅ 断言检查通过！")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n--- 测试脚本执行完毕 ---")

if __name__ == "__main__":
    main()
    