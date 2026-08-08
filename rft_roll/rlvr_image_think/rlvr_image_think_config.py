import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any

from roll.configs.base_config import BaseConfig, ScheduleConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

from roll.pipeline.rlvr.rlvr_config import RLVRConfig, RewardConfig


@dataclass
class ImageEditRewardConfig(RewardConfig):
    """
    继承自 RewardConfig, 并为图像编辑任务增加了特定的模型路径。
    """  
    clusters: Optional[List[Dict[str, Any]]] = field(
        default=None, 
        metadata={"help": "A list of cluster configurations for multi-model workers."}
    )
    edit_max_area: int = 1024 * 1024
    edit_sampling_steps: int = 40
    edit_is_think_model: bool = False
    
    is_offload_state_edit_model: bool = True    # 是否卸载 edit model to CPU
    is_offload_state_vlm: bool = True           # 是否卸载 vlm model to CPU
    
    
@dataclass
class RLVRImageThinkConfig(RLVRConfig):
    """
    继承自 RLVRConfig, 并指定 rewards 字段使用扩展后的 ImageEditRewardConfig。
    """
    DEFAULT_WORKER_CLS: str = "rlvr_image_think.rlvr_actor_worker.ImageThinkActorWorker"
    
    # 覆盖父类的 rewards 字段
    rewards: Optional[Dict[str, ImageEditRewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for the multi domain rewards, using extended ImageEditRewardConfig."}
    )
    
    cot_prompt_version: str = "reason_edit"  # vlm -> cot 使用哪个 system prompt

    def __post_init__(self):
        super().__post_init__()
    
        # reset default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = self.DEFAULT_WORKER_CLS
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = self.DEFAULT_WORKER_CLS
        if self.reference.worker_cls is None:
            self.reference.worker_cls = self.DEFAULT_WORKER_CLS
