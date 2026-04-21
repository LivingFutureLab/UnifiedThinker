# coding=utf--8
# jianchong.zq: 本地单步调试


import os, sys 
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
rolldir = os.path.join(rootdir, "src_rl")
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
if rolldir not in sys.path:
    sys.path.insert(0, rolldir)

import copy
import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib

from concurrent import futures
from typing import List
import tempfile
import datasets
import PIL.Image as Image
import ray
import torch
from codetiming import Timer
from datasets import load_dataset, load_from_disk
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer
from transformers import AutoConfig, ProcessorMixin
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from codetiming import Timer
from transformers import set_seed

from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracking import create_tracker
from roll.utils.worker_state import WorkerState

from roll.datasets.collator import DataCollatorWithPaddingForMM
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from rlvr_image_think.rlvr_image_think_config import RLVRImageThinkConfig
from roll.pipeline.rlvr.rlvr_pipeline import query_filter_fn, update_dataset_domain
from roll.utils.checkpoint_manager import download_model
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_advantage,
    compute_token_reward,
    get_sample_level_mask,
    reduce_metrics,
    reward_postprocess,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager
from roll.utils.packages import is_transformers_version_greater_than

logger = get_logger()

from rlvr_image_think.rlvr_image_think_pipeline import (format_prompt, process_image, get_vlm_dataset,
                                            RLVRImageThinkConfig, encode_function)


def import_class_from_string(path: str) -> Any:
    """
    Dynamically imports a class from a string path.
    e.g., "roll.worker.generation_worker.GenerationWorker" -> <class 'roll.worker.generation_worker.GenerationWorker'>
    """
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Could not import class from string '{path}'") from e

# ==============================================================================
# 创建一个可调试的 Pipeline 类
# ==============================================================================
class DebuggableRLVRImageThinkPipeline(BasePipeline):
    """
    这是一个修改后的版本，用于单步调试，同时继承并遵循 BasePipeline 的结构。

    - 继承自 BasePipeline, 复用其状态管理、checkpoint 和 tracker 功能。
    - 移除了 Ray Actor 和 Cluster, 直接在本地实例化 Worker 类。
    - 所有远程调用 (.remote(), ray.get()) 被替换为直接的方法调用。
    - 重写了 `do_checkpoint` 方法，使其在单进程模式下工作。
    - `run` 方法被实现为一个简单的循环，每次调用 `debug_step`。
    - 在 `debug_step` 方法中可以轻松设置断点。
    """
    def __init__(self, pipeline_config: RLVRImageThinkConfig):
        #super().__init__(pipeline_config)  # 绕过 ResourceManager
        print("--------- Custom Init: Bypassing ResourceManager, following src_rl/roll/pipeline/base_pipeline.py ---------")
        set_seed(seed=pipeline_config.seed)
        self.pipeline_config = pipeline_config
        
        # 绕过 ResourceManager
        self.resource_manager = None
        self.model_update_groups: List[ModelUpdateGroup] = []
        self.checkpoint_clusters: List = []
        logger.warning("ResourceManager has been bypassed for single-process debugging.")
        
        # 手动实现 BasePipeline 的其余初始化逻辑
        self.state = WorkerState()
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.pipeline_config.checkpoint_config)
        self.tracker = create_tracker(
            tracker_name=self.pipeline_config.track_with,
            config=self.pipeline_config.to_dict(),
            **self.pipeline_config.tracker_kwargs,
        )
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        
        # 手动处理从 checkpoint 恢复的逻辑
        self.resume_futures = []
        self.resume_from_checkpoint = False
        if self.pipeline_config.resume_from_checkpoint:
            self.resume_from_checkpoint = download_model(self.pipeline_config.resume_from_checkpoint)
            logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
            load_dir = os.path.join(self.resume_from_checkpoint, "pipeline")
            self.state = WorkerState.load_from_json(load_dir=load_dir, tag="pipeline")

            def resume_metrics():
                for metrics in self.state.log_history:
                    self.tracker.log(values=metrics, step=metrics["system/step"])
            self.resume_futures.append(self.executor.submit(resume_metrics))
            
        print("--------- Custom Init: Bypassing ResourceManager, following src_rl/roll/pipeline/base_pipeline.py ---------")
            
        
        # --- 初始化 Processor 和 Tokenizer ---
        self.processor: ProcessorMixin = default_processor_provider(self.pipeline_config.actor_train.model_args)
        # ... (省略了 image_processor 的 max_pixels 设置，与原始代码相同)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # --- 3. 加载和预处理数据集 (与原始代码相同) ---
        print("Loading and preparing dataset...")
        dataset = get_vlm_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, get_eval=False
        )
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
        )
        self.domain_datasets: Dict[str, datasets.Dataset] = {
            domain: dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys()
        }

        # --- 4. [MODIFIED] 直接实例化 Worker，而不是创建 Ray Cluster ---
        print("Instantiating workers locally...")
        # 注意：worker的构造函数可能需要一个 resource_manager，这里我们传入 None 或一个虚拟的实例
        # 这里的 None 对应原始代码中 Ray Cluster 的 resource_manager 参数
        ActorTrainWorker = import_class_from_string(self.pipeline_config.actor_train.worker_cls)
        self.actor_train: Any = ActorTrainWorker(self.pipeline_config.actor_train)
        
        ActorInferWorker = import_class_from_string(self.pipeline_config.actor_infer.worker_cls)
        self.actor_infer: Any = ActorInferWorker(self.pipeline_config.actor_infer)
        
        ReferenceWorker = import_class_from_string(self.pipeline_config.reference.worker_cls)
        self.reference: Any = ReferenceWorker(self.pipeline_config.reference)
        
        if self.pipeline_config.adv_estimator == "gae" and hasattr(self.pipeline_config, 'critic'):
            CriticWorker = import_class_from_string(self.pipeline_config.critic.worker_cls)
            self.critic: Any = CriticWorker(self.pipeline_config.critic)

        self.rewards: Dict[str, Any] = {}
        for key, worker_config in self.pipeline_config.rewards.items():
            RewardWorker = import_class_from_string(worker_config.worker_cls)
            self.rewards[key] = RewardWorker(worker_config)
            
        # ==============================================================================
        # 手动为需要 GPU 资源的 Worker 设置 resource_placement_groups。
        # 这模拟了 ResourceManager 在分布式环境中的行为。
        # 对于本地单 GPU 调试，通常就是 [{"GPU": 1}]。
        # ==============================================================================
        print("Manually setting resource placement groups for local debugging...")
        single_gpu_placement_group = [{"GPU": 1}]

        # actor_infer, reference, 和 rewards 通常使用 VLLMStrategy，需要这个配置。
        # 检查 worker_config 是否有该属性，以增加代码的健壮性。
        if hasattr(self.actor_infer.worker_config, "resource_placement_groups"):
            self.actor_infer.worker_config.resource_placement_groups = single_gpu_placement_group
            
        if hasattr(self.reference.worker_config, "resource_placement_groups"):
            self.reference.worker_config.resource_placement_groups = single_gpu_placement_group

        single_gpu_placement_group = [{"GPU": 2}]
        for key, worker in self.rewards.items():
            if hasattr(worker.worker_config, "resource_placement_groups"):
                worker.worker_config.resource_placement_groups = single_gpu_placement_group

        # actor_train 和 critic 通常使用 DeepSpeed 或其他训练策略，可能不需要这个配置，
        # 但为了安全起见，可以同样检查并设置。
        single_gpu_placement_group = [{"GPU": 3}]
        if hasattr(self.actor_train.worker_config, "resource_placement_groups"):
            self.actor_train.worker_config.resource_placement_groups = single_gpu_placement_group

        if hasattr(self, 'critic') and hasattr(self.critic.worker_config, "resource_placement_groups"):
            self.critic.worker_config.resource_placement_groups = single_gpu_placement_group

        # --- 5. [MODIFIED] 直接初始化所有 Worker ---
        print("Initializing workers...")
        self.actor_infer.initialize(pipeline_config=self.pipeline_config)
        self.reference.initialize(pipeline_config=self.pipeline_config)
        for key, worker in self.rewards.items():
            worker.initialize(pipeline_config=self.pipeline_config)
        self.actor_train.initialize(pipeline_config=self.pipeline_config)
        if self.pipeline_config.adv_estimator == "gae":
            self.critic.initialize(pipeline_config=self.pipeline_config)
        
        # --- 6. [NEW] 设置模型更新和 checkpointing ---
        # 使用 BasePipeline 的方法来注册需要同步和保存的模型
        self.set_model_update_pair(
            src_cluster=self.actor_train, # 虽然是本地对象，但接口兼容
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        # --- 7. [NEW] 手动创建数据整理器 (Data Collator) ---
        self.collator = DataCollatorWithPaddingForMM(
            processor=self.processor,
            extra_unpadded_keys=["domain", "reward_model"],
            extra_data_provider=get_extra_data_provider(
                self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
            ),
            prompt_key="prompt", answer_key="ground_truth", image_key="images",
            max_length=self.pipeline_config.prompt_length, padding="max_length",
        )
        
        # --- 8. 初始化 RL 相关组件 (与原始代码相同) ---
        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )
        self.running = {domain: RunningMoments() for domain in self.rewards.keys()}
        print("Debuggable Pipeline Initialized Successfully!")


    def run(self):
        """
        [NEW] 实现 BasePipeline 的 run 方法。
        这是一个简单的循环，用于按顺序执行 PPO 步骤。
        它会从 `self.state.step` (由 checkpoint 加载) 开始。
        """
        print(f"Starting run from step: {self.state.step}")
        for step in range(self.state.step, self.pipeline_config.max_steps):
            self.global_step = step
            self.debug_step() # 执行单步逻辑
        print("Pipeline run completed!")

    @torch.no_grad()
    def debug_step(self):
        """
        [MODIFIED] 这是执行 PPO 算法单个完整步骤的函数。
        它对应于原始 `run` 方法中的 for 循环的一次迭代。
        在这里设置断点 (`import pdb; pdb.set_trace()`) 来进行调试。
        """
        logger.info(f"--- Starting Debug Step {self.global_step} ---")
        
        # <<< 在这里设置断点来开始调试 >>>
        import pdb; pdb.set_trace()
        
        # === 1. 模型同步更新 ===
        # 使用 BasePipeline 的方法
        with Timer(name="model_update", logger=None) as t:
            self.model_update(self.global_step)
        logger.info(f"Model update took {t.last:.2f}s")
        
        # === 2. 数据生成（Experience Generation）===
        # (这部分逻辑与上一个版本的 debug_step 相同，只是 GPU 管理更明确)
        logger.info("Generating experience...")
        self.actor_infer.load_states()
        for reward_worker in self.rewards.values():
            reward_worker.load_states()
        
        domain_batches = []
        # ... (省略了从数据集中采样和组合批次的逻辑，与上一个版本完全相同)
        
        batch = DataProto.concat(domain_batches)

        self.actor_infer.offload_states()
        for reward_worker in self.rewards.values():
            reward_worker.offload_states()
        logger.info("Experience generation complete.")
        
        # === 3. 计算 Ref Log Probs, Old Log Probs & Values ===
        # (这部分逻辑也与上一个版本相同，但加入了更明确的 GPU 状态管理)
        logger.info("Computing reference log probs...")
        self.reference.load_states()
        ref_log_probs = self.reference.compute_log_probs(batch)
        self.reference.offload_states()
        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
        batch = batch.union(ref_log_probs)

        logger.info("Computing old policy log probs and values...")
        self.actor_train.load_states()
        if self.pipeline_config.adv_estimator == "gae":
            self.critic.load_states()

        old_log_probs = self.actor_train.compute_log_probs(batch)
        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
        if self.pipeline_config.adv_estimator == "gae":
            values = self.critic.compute_values(batch)
            batch = batch.union(values)

        # === 4. 计算优势和训练 ===
        # (这部分逻辑也基本相同)
        logger.info("Post-processing rewards and computing advantages...")
        # ... (省略了 advantage 计算的完整代码块，它和上一个版本相同)
        
        logger.info("Executing training step...")
        if self.pipeline_config.adv_estimator == "gae":
            self.critic.train_step(batch)

        if self.pipeline_config.critic_warmup <= self.global_step:
            self.actor_train.train_step(batch)

        # 训练结束后，可以卸载训练模型以节省显存（如果在循环中需要）
        self.actor_train.offload_states()
        if self.pipeline_config.adv_estimator == "gae":
            self.critic.offload_states()

        # === 5. [NEW] 更新状态和执行 Checkpoint ===
        # 使用 BasePipeline 的机制
        self.state.step = self.global_step
        
        # 伪造一个 metrics 字典用于保存状态，实际应用中应收集真实 metrics
        fake_metrics = {"system/step": self.global_step, "loss/actor": 0.1, "reward/mean": 1.0}
        self.state.log_history.append(fake_metrics)
        self.tracker.log(values=fake_metrics, step=self.global_step)
        
        self.do_checkpoint(global_step=self.global_step)
        
        logger.info(f"--- Debug Step {self.global_step} Finished ---\n")

    def do_checkpoint(self, global_step):
        """
        [OVERRIDE] 重写 BasePipeline 的 do_checkpoint 方法以适应单进程调试。
        这里没有 Ray ObjectRefs，所以我们直接进行阻塞调用。
        """
        # 从 state 中获取最新的 metrics
        metrics = self.state.log_history[-1] if self.state.log_history else {}
        metrics["system/step"] = global_step

        if global_step > 0 and (
            global_step % self.pipeline_config.save_steps == 0 or global_step == self.pipeline_config.max_steps - 1
        ):
            logger.info(f"Saving checkpoint for step {global_step}...")
            
            # [MODIFIED] 直接调用 worker 的 do_checkpoint 方法
            for worker in self.checkpoint_clusters:
                # 调用是阻塞的，直接返回 DataProto 对象
                ckpt_metrics_proto = worker.do_checkpoint(global_step=global_step, blocking=True)
                if ckpt_metrics_proto and ckpt_metrics_proto.meta_info.get("metrics"):
                    metrics.update(reduce_metrics(ckpt_metrics_proto.meta_info.pop("metrics", {})))

            # 之后的文件保存逻辑与 BasePipeline 相同
            ckpt_id = f"checkpoint-{global_step}"
            pipeline_save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
            save_dir = os.path.join(pipeline_save_dir, "pipeline")
            
            self.state.save_to_json(save_dir=save_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
            
            # CheckpointManager 的 upload 可能会与云存储交互
            # 在本地调试时，你可以注释掉它或确保配置正确
            try:
                self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)
                logger.info(f"Checkpoint {ckpt_id} saved and uploaded.")
            except Exception as e:
                logger.warning(f"Failed to upload checkpoint: {e}")

        # 这部分处理 resume 的 future，保持不变
        futures.wait(self.resume_futures)
        self.resume_futures.clear()


if __name__ == "__main__":
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    from dacite import from_dict, Config as DaciteConfig
    
    import pdb; pdb.set_trace()

    # 要加载的主配置文件名 (不带 .yaml 后缀)
    CONFIG_NAME = "rlvr_image_think_debug"  

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
   

    # 为了调试，可以强制减少批次大小和数据集大小
    DEBUG = True
    pipeline_config.rollout_batch_size = 4
    pipeline_config.actor_train.train_batch_size = 2
    pipeline_config.max_steps = 5 # 运行几个步骤进行测试
    pipeline_config.save_steps = 2 # 测试 checkpointing

    # Ray 在这里不是必需的，但如果某些底层库隐式调用了它，local_mode 可以防止出错
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)

    # 2. 实例化可调试的 Pipeline
    debug_pipeline = DebuggableRLVRImageThinkPipeline(pipeline_config)
    
    # 3. 运行 Pipeline
    print("\n=======================================================")
    print("Setup complete. You can now start debugging.")
    print("To debug, place `import pdb; pdb.set_trace()` inside the")
    print("`debug_step` method and then run this script.")
    print("=======================================================\n")
    
    # 调用 `run()` 将启动训练循环
    # debug_pipeline.run()
    
    # 或者，如果你想手动控制每一步：
    print("Executing a single step manually...")
    import pdb; pdb.set_trace()
    debug_pipeline.global_step = debug_pipeline.state.step
    debug_pipeline.debug_step()
    
    print("\nManual step finished. You can call `debug_pipeline.debug_step()` again in the pdb console.")
    