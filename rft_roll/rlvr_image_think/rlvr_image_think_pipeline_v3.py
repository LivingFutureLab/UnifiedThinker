#coding=utf-8
#jianchong.zq: 适配 tbstar-image-think
""" 
主要修改:
    在训练过程中将 actor_train (使用 megatron_train 策略) 的模型参数同步到 reward worker (使用 hf_infer 策略) 的某个 module。
"""

import os, sys    
import copy
import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent import futures

import concurrent.futures
from functools import partial
import contextlib
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

from roll.datasets.collator import DataCollatorWithPaddingForMM
#from roll.datasets.dataset import get_dataset
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
#from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from rlvr_image_think.rlvr_image_think_config import RLVRImageThinkConfig
from roll.pipeline.rlvr.rlvr_pipeline import query_filter_fn, update_dataset_domain
#from roll.utils.checkpoint_manager import download_model
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

from rlvr_image_think.image_edit_think_pipe.util import preprocess_image
from rlvr_image_think.utils import download_model_mos, pre_download_and_update_dataset, pre_download_and_load_images

logger = get_logger()

def format_prompt(prompt, processor, prompt_image_token="<image>"):
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", 
                             "text": "你是一位专业的视觉推理专家和图像编辑顾问。你的核心任务是：\n1.  **深入分析**: 接收用户提供的原始图片和编辑指令（edit prompt）。\n2.  **逻辑推理**: 结合图片内容（如物体材质、所处环境、当前状态）和生活常识、物理规律或特定艺术风格等，对编辑指令进行深度推理，预测出指令执行后最可能产生的视觉结果。\n3.  **精准描述**: 将推理出的编辑后图像样貌，用一段精炼、客观、富有画面感的文字描述出来。\n\n要求：\n- 直接返回最终的图像内容描述。\n- 描述内容控制在200字以内。\n- 禁止包含任何解释、分析过程或多余的客套话。"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],  # image_token has been included in prompt
            }
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def process_image(image: Image.Image, processor: ProcessorMixin):
    # same as qwen2-vl image processor
    image_processor = processor.image_processor
    factor = (
        image_processor.patch_size * image_processor.merge_size
        if "Qwen" in image_processor.image_processor_type
        else 28
    )
    height, width = image.height, image.width
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=image_processor.min_pixels,
        max_pixels=image_processor.max_pixels,
    )
    resized_image = image.resize((resized_width, resized_height), resample=image_processor.resample)
    return resized_image

def process_images(
    images: Union[List, Tuple, str, Image.Image], processor: ProcessorMixin
) -> Union[Image.Image, List[Image.Image], List[List[Image.Image]]]:
    """Process images, handling different levels of nesting.

    Args:
      images: A single image, a list of images, or a list of lists of images to load.
      timeout: Timeout for loading images.

    Returns:
      A single image, a list of images, a list of lists of images.
    """
    if isinstance(images, (list, tuple)):
        if len(images) and isinstance(images[0], (list, tuple)):
            return [[process_image(image, processor=processor) for image in image_group] for image_group in images]
        else:
            return [process_image(image, processor=processor) for image in images]
    else:
        return process_image(images, processor=processor)

oss_config = {
    "bucket_name": bucket_name,
    "access_id": oss_access_id,
    "access_key": oss_access_key,
    "endpoint": oss_endpoint
}

def encode_function(
    data, processor, prompt_getter, image_getter, tag_getter, 
    edit_prompt_getter, edit_prompt_cot_getter,
    prompt_image_token="<image>"
):
    # 批处理
    image_flag = [True] * len(prompt_getter(data))
    image_list = []
    pre_edit_image_list = []    # used for image-edit-reward-model
    for idx, image_paths in enumerate(image_getter(data)):
        if not image_paths:   # None 或空列表/元组的情况
            image_flag[idx] = False
        
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        try:
            # ## 方案1:
            # with tempfile.TemporaryDirectory() as temp_dir:
            #     # local_files = [os.path.join(temp_dir, f"{i}_{os.path.basename(p)}") for i, p in enumerate(image_paths)]
            #     # for oss_path, local_path in zip(image_paths, local_files):
            #     #     oss_path = oss_path.replace(f"oss://{bucket_name}/", "")
            #     #     oss_bucket.get_object_to_file(oss_path, local_path)  
            #     image_out = load_images(local_files, timeout=None)
            
            # ## 方案二: image_paths 是一个只包含有效本地路径的列表: call pre_download_and_update_dataset first
            # existing_files = [p for p in image_paths if os.path.exists(p)]
            # if len(existing_files) != len(image_paths):
            #     logger.warning(f"Some local image paths not found. Expected: {len(image_paths)}, Found: {len(existing_files)}")
            # if not existing_files:
            #     image_out = []
            # else:
            #     image_out = load_images(existing_files, timeout=None)
            
            ## 方案三: image_paths 已经是 PIL.Image, call pre_download_and_load_images first
            image_out = image_paths
                
        except Exception as e:
            # 兜底逻辑
            image_out = [Image.new("RGB", (224, 224), (255, 255, 255))] * len(image_paths)
            logger.error(f"Failed to get image: {image_paths}, error of {str(e)}")
        
        # resize image following qwen-image-edit-think first
        image_out = [preprocess_image(im, max_area=1024 * 1024, adjust_ar=False) for im in image_out]
        pre_edit_image_list.append(copy.deepcopy(image_out))
        
        image_out = process_images(image_out, processor)
        image_list.append(image_out)
        
    text_list = []
    for idx, instruct in enumerate(prompt_getter(data)):
        # add prompt_image_token to prompt
        img_prompt_template = "Picture {number}: {token}"
        base_img_prompt = ""
        if image_flag[idx]:
            imgs = image_list[idx]
            if isinstance(imgs, list):
                base_img_prompt = ""
                for i, img in enumerate(imgs):
                    base_img_prompt += img_prompt_template.format(number=i + 1, token=prompt_image_token)
            elif imgs is not None:
                base_img_prompt = img_prompt_template.format(number=1, token=prompt_image_token)
        
        instruct = instruct.replace(prompt_image_token, "").strip()
        instruct = base_img_prompt + instruct                    
        text = format_prompt(instruct, processor, prompt_image_token)
        # if DEBUG:
        #     print("[Debug] text\n {}".format(json.dumps(text, ensure_ascii=False)))
        
        text_list.append(text)
    encodings = {
        "tag": tag_getter(data),
        "images": image_list,
        "prompt": text_list,

        "pre_edit_images": pre_edit_image_list,
        "edit_prompt": edit_prompt_getter(data),
        "edit_prompt_cot": edit_prompt_cot_getter(data),
        
    }
    return encodings

def get_vlm_dataset(data_args, encode_function, processor, get_eval=False, debug_mode=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset

    #dataset = get_dataset(data_args=data_args)
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    FILEEXT2TYPE = {
        "arrow": "arrow",
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "parquet": "parquet",
        "txt": "text",
    }
    if isinstance(data_name, list):
        local_path = ""
    else:
        local_path: str = os.path.join(dataset_dir, data_name)
    if os.path.isdir(local_path):
        for file_name in os.listdir(local_path):
            data_files.append(os.path.join(local_path, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
        data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
    else:
        assert local_path == ""
        for file_name in data_name:
            data_files.append(os.path.join(dataset_dir, file_name))
            if data_path is None:
                data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
            elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                raise ValueError("File types should be identical.")
    dataset = load_dataset(path=data_path, data_files=data_files)["train"]
    if debug_mode:
        logger.info("Debug 模式, 只使用 1000 样本")
        dataset = dataset.shuffle(seed=0).select(range(256))
    # if True:
    #     logger.info("debug， 只使用 1000 样本")
    #     dataset = dataset.shuffle(seed=0).select(range(256))
        
    # regularized data filed
    features = datasets.Features(
        {
            "tag": datasets.Value(dtype="string"),  
            "images": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)), 
            "prompt": datasets.Value(dtype="string"),                   # used for actor
            "domain": datasets.Value(dtype="string"),  
            
            # the following is used for image-edit-reward
            "pre_edit_images": datasets.Sequence(feature=datasets.Image(mode=None, decode=True)), 
            "edit_prompt": datasets.Value(dtype="string"),             
            "edit_prompt_cot": datasets.Value(dtype="string"),          # used for reward model, high quality from gemini2.5-pro
        }
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    
    prompt_getter = lambda data: data["edit_prompt"]
    image_getter = lambda data: data["ref_imgs"]
    tag_getter = lambda data: data["task_type"]
    # the following used for image-edit-reward-model
    edit_prompt_getter = lambda data: data["edit_prompt"]
    edit_prompt_cot_getter = lambda data: data.get("edit_prompt_cot", data['reference_text'])
    
    # 预下载所有图片
    # dataset = pre_download_and_update_dataset(dataset, image_column="ref_imgs", cache_path="./cache/image_download/",
    #                                           oss_config=oss_config, num_proc=32)
    dataset = pre_download_and_load_images(dataset, image_column="ref_imgs",
                                              oss_config=oss_config, num_proc=32)
    
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(
            data, processor, prompt_getter, image_getter, tag_getter, 
            edit_prompt_getter, edit_prompt_cot_getter, 
            prompt_image_token="<image>"
        ),
        batched=True,      
        batch_size=32,
        num_proc=data_args.preprocessing_num_workers,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset

def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model_mos(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature  # help define a object to accesss attr

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature({"spatial_merge_size": processor.image_processor.merge_size}),
                        "image_token_id": processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                        "video_token_id": processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    }
                )
            }
        )
        if is_transformers_version_greater_than("4.52.0"):
            from transformers.models.qwen2_vl import Qwen2VLModel

            get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)
        else:
            from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

            get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)[0]
            # (3, bsz, seqlen) -> (bsz, 3, seqlen) to put it into DataProto,
            # transpose it batck to (3, bsz, seqlen) before forward for model
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider

    def default_extra_data_provider(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {"position_ids": position_ids}

    return default_extra_data_provider


class RLVRImageThinkPipeline(BasePipeline):
    def __init__(self, pipeline_config: RLVRImageThinkConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        
        self.debug_mode = False
        if self.pipeline_config.system_envs.get("RAY_DEBUG", "") == "legacy":
            self.debug_mode = True
            #import pdb; pdb.set_trace()
            breakpoint()    # ray debugger using 'ray start --head'
            logger.info("-- 进入调试 --")

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)        # e.g., <class 'transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor'>
        # set max_pixels to avoid image token num is larger than prompt length
        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.actor_train.model_args, "max_pixels", 1024 * 1024),
            getattr(self.pipeline_config.actor_train.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        
        # --- 2. 加载和预处理数据集 ---
        dataset = get_vlm_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, get_eval=False, debug_mode=self.debug_mode
        )
        # update domain field, DynamicSamplingScheduler requires
        dataset = dataset.map(
            partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
            num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
            desc="update_dataset_domain",
            load_from_cache_file=False,
        )

        # --- 3. 按领域 (Domain) 拆分数据集 ---
        self.domain_datasets: Dict[str, datasets.Dataset] = {}
        for domain in self.pipeline_config.actor_train.data_args.domain_interleave_probs.keys():
            # # 使用 .filter() 方法，从总数据集中筛选出属于当前领域的数据
            self.domain_datasets[domain] = dataset.filter(
                lambda example, dom: example["domain"] == dom,
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                fn_kwargs={"dom": domain},
            )
            assert len(self.domain_datasets[domain]) > 0, f"domain dataset {domain} has no data"

        # --- 4. 加载验证集 (如果配置了) ---
        self.val_dataset = None
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            self.val_dataset = get_vlm_dataset(
                self.pipeline_config.validation.data_args, encode_function, self.processor, get_eval=True
            )
            self.val_dataset = self.val_dataset.map(
                partial(update_dataset_domain, self.pipeline_config.tag_2_domain),
                num_proc=self.pipeline_config.actor_train.data_args.preprocessing_num_workers,
                desc="update_val_dataset_domain",
                load_from_cache_file=False,
            )

        # --- 5. 初始化 RL 相关组件 ---
        # 初始化 KL 散度控制器。在 PPO 等算法中，KL 惩罚用于防止策略（Actor）模型在更新后与参考模型偏离太远。
        # 这个控制器会根据当前 KL 值动态调整惩罚系数 (kl_coef)。
        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        assert self.pipeline_config.max_steps > 0, "max_steps must be greater than 0"
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        # --- 6. 创建分布式组件集群 (Clusters) ---
        # Cluster 是对一组 Ray Actor 的封装，代表一个功能单元。
        
        # Actor (训练) 集群：负责执行模型参数的梯度更新。
        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        # Actor (推理) 集群：负责使用当前策略生成响应（"rollouts"）。与训练分离可以提高效率。
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        # Reference (参考) 模型集群：持有一个固定的、训练前的模型副本，用于计算 KL 散度。
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        # Critic (价值) 模型集群：如果使用 GAE (Generalized Advantage Estimation) 作为优势估计器，
        # 则需要一个 Critic 模型来评估状态的价值 V(s)。
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
        # Reward (奖励) 模型集群：一个字典，键是领域名，值是对应领域的奖励模型集群。
        # 这种设计允许为不同的数据领域使用不同的奖励模型。
        # key must be same as domain, which is used in DynamicSamplingScheduler
        # to get corresponding reward
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        
        # --- 7. 初始化动态采样调度器 (DynamicSamplingScheduler) ---
        # 这是整个 RL 经验生成（rollout）阶段的核心协调者。

        # 获取各领域的采样比例
        domain_ratios = self.pipeline_config.actor_train.data_args.domain_interleave_probs
        self.generate_schedulers: Dict[str, DynamicSamplingScheduler] = {}      # 存储每个领域的调度器
        self.domain_batch_size = {}                                             # 存储每个领域的 rollout batch size
        domain_list = list(domain_ratios.keys())
        accumulated = 0
        # 遍历每个领域，为其创建一个专属的调度器
        for i, domain in enumerate(domain_list):
            if i == len(domain_list) - 1:
                domain_batch_size = self.pipeline_config.rollout_batch_size - accumulated
            else:
                domain_batch_size = int(domain_ratios[domain] * self.pipeline_config.rollout_batch_size)
            accumulated += domain_batch_size
            
            # 创建一个 DynamicSamplingScheduler 的 Ray Actor 实例。
            # NodeAffinitySchedulingStrategy 确保这个调度器 Actor 运行在主进程所在的节点上，以减少通信开销。
            generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(pipeline_config=self.pipeline_config)
            
            # 远程调用 set_scheduler 方法，对这个调度器进行详细配置
            ray.get(    # ray.get() 会阻塞直到远程调用完成
                generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,                             # 指定用于生成内容的 Actor 集群
                    reward_clusters={domain: self.rewards[domain]},             # 指定用于打分的奖励模型集群 (只用当前领域的)
                    dataset=self.domain_datasets[domain],                       # 指定该调度器使用的数据集
                    collect_fn_cls=DataCollatorWithPaddingForMM,                # 指定数据整理器 (collator)，用于批处理多模态数据
                    collect_fn_kwargs=dict(
                        # tokenizer passed by DynamicSamplingScheduler.set_scheduler
                        # tokenizer=self.tokenizer,
                        processor=self.processor,
                        extra_unpadded_keys=["domain", "pre_edit_images", "edit_prompt", "edit_prompt_cot"],         # 指定哪些元数据字段不需要填充
                        extra_data_provider=get_extra_data_provider(
                            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
                        ),                                                      # 获取额外数据的辅助函数
                        prompt_key="prompt",
                        #answer_key="ground_truth",
                        answer_key=None,
                        image_key="images",
                        image_flag_key=None,
                        max_length=self.pipeline_config.prompt_length,
                        padding="max_length",
                    ),
                    response_filter_fn=lambda data_item, config: True,          # 响应过滤器，这里不过滤
                    query_filter_fn=query_filter_fn,
                    # 设置回调函数：当生成一个响应并获得奖励后，调用此函数。
                    # 这里是把结果报告给调度器自己，用于收集和缓冲。
                    response_callback_fn=generate_scheduler.report_response.remote,
                    state=self.state.kv.get(f"scheduler_state_{domain}", None),
                )
            )
            self.generate_schedulers[domain] = generate_scheduler
            self.domain_batch_size[domain] = domain_batch_size

            assert domain_batch_size < len(self.domain_datasets[domain]), (
                f"domain_batch_size {domain_batch_size} must be "
                f"less than the number of domain datasets {len(self.domain_datasets[domain])}"
            )

        # ---- 8. 初始化验证集的调度器 ----
        if self.val_dataset:
            val_pipeline_config = copy.deepcopy(self.pipeline_config)
            val_pipeline_config.is_use_additional_prompts = False
            self.val_generate_scheduler = DynamicSamplingScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(pipeline_config=val_pipeline_config)
        if self.val_dataset:
            ray.get(
                self.val_generate_scheduler.set_scheduler.remote(
                    actor_cluster=self.actor_infer,
                    reward_clusters=self.rewards,
                    dataset=self.val_dataset,
                    collect_fn_cls=DataCollatorWithPaddingForMM,
                    collect_fn_kwargs=dict(
                        # tokenizer passed by DynamicSamplingScheduler.set_scheduler
                        # tokenizer=self.tokenizer,
                        processor=self.processor,
                        # val metrics are grouped by tag rather than domain
                        extra_unpadded_keys=["domain", "reward_model", "tag"],
                        extra_data_provider=get_extra_data_provider(
                            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
                        ),
                        prompt_key="prompt",
                        #answer_key="ground_truth",
                        answer_key=None,
                        image_key="images",
                        image_flag_key=None,
                        max_length=self.pipeline_config.prompt_length,
                        padding="max_length",
                    ),
                    response_filter_fn=lambda data_item, config: True,
                    query_filter_fn=lambda data_list, config: True,
                    response_callback_fn=self.val_generate_scheduler.report_response.remote,
                )
            )

        # --- 9. 启动并初始化所有分布式集群 ---
        # 这是一个异步启动过程，可以提高启动效率。
        refs = []
        # 非阻塞地初始化 actor_infer 集群
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)   # 等待 actor_infer 初始化完成

        # 阻塞地初始化 reference 集群，因为后续组件可能依赖它
        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        
        # 非阻塞地初始化所有 reward 模型集群
        refs = []
        for key, cluster in self.rewards.items():
            refs.extend(cluster.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        
        # 非阻塞地初始化 actor_train 和 critic 集群
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        # --- 10. 设置模型同步和检查点 ---
        # 设置一个同步对：定期将 actor_train 的模型权重复制到 actor_infer。这确保了用于生成经验的策略模型能及时更新。
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )
        
        # 建立 actor_train 到 reward worker 的同步关系
        self.set_model_update_pair(  
            src_cluster=self.actor_train,  
            tgt_cluster=self.rewards['reason_edit'],
            frequency=1  # 同步频率  
        )
        

        # 指定在保存检查点 (checkpoint) 时需要保存哪些模型的权重。
        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        # --- 11. 初始化其他状态 ---
        self.running = {}
        for domain in self.rewards.keys():
            self.running[domain] = RunningMoments()
    
    # def do_checkpoint(self, global_step):
    #     metrics = self.state.log_history[-1]
    #     metrics["system/step"] = global_step
    #     if global_step > 0 and (
    #         global_step % self.pipeline_config.save_steps == 0 or global_step == self.pipeline_config.max_steps - 1
    #         or global_step == 1 # jianchong.zq: step-1 验证性保存一次 ckpt
    #     ):
    #         # ckpt_metrics_refss = []
    #         # for cluster in self.checkpoint_clusters:
    #         #     ckpt_metrics_refss.append(cluster.do_checkpoint(global_step=global_step, blocking=False))

    #         # for ckpt_metrics_refs in ckpt_metrics_refss:
    #         #     ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
    #         #     metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))
            
    #         # 解决并行执行do_checkpoint, 导致内存爆炸
    #         for cluster in self.checkpoint_clusters:
    #             # 1. 直接调用，让它阻塞，等待完成
    #             ckpt_metrics_refs = cluster.do_checkpoint(global_step=global_step, blocking=True) # 或者移除 blocking 参数，如果默认是 True
    #             # 2. 立即处理结果
    #             ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
    #             metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))
            
    #         ckpt_id = f"checkpoint-{global_step}"
    #         pipeline_save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id)
    #         save_dir = os.path.join(self.pipeline_config.output_dir, "pipeline", ckpt_id, "pipeline")
    #         self.state.save_to_json(save_dir=save_dir, tag="pipeline")
    #         self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
    #         self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)

    #     futures.wait(self.resume_futures)
    #     self.resume_futures.clear()

    @torch.no_grad()
    def run(self):
        if self.debug_mode:
            #import pdb; pdb.set_trace()
            breakpoint()    # ray debugger using 'ray start --head'
            logger.info("-- 进入调试 --")
            
        metrics_mgr = MetricsManager()

        # 初始化多个计时器，用于监控不同阶段的性能
        tps_timer = _Timer(window_size=5)
        actor_infer_timer = _Timer(window_size=5)
        actor_infer_response_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            self.global_step = global_step
            logger.info(f"pipeline step {global_step} start...")

            # 在每个新步骤开始时，清空上一轮的度量
            metrics_mgr.clear_metrics()
            with tps_timer, Timer(name="step_total", logger=None) as step_total_timer:
                
                # === 内存管理：为生成阶段准备GPU显存 ===
                # 将训练模型（Critic 和 Actor-Train）的权重从GPU卸载到CPU，以释放GPU显存
                # 这是因为接下来的数据生成阶段需要加载推理模型(actor_infer)和奖励模型(rewards)。

                # GAE (Generalized Advantage Estimation) 算法需要 Critic 模型，所以也要卸载。
                if self.pipeline_config.adv_estimator == "gae":     # grpo
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                # === 2. 模型同步更新 ===
                # 从一个参数服务器或者主节点同步最新的模型权重，确保所有工作节点模型一致 ?
                with Timer(name="step_model_update", logger=None) as step_model_update_timer:
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics_mgr.add_metrics(model_update_metrics)
                    metrics_mgr.add_metric("time/step_model_update", step_model_update_timer.last)

                # === 3. 周期性评估（Validation） ===
                if self.val_dataset and global_step % self.pipeline_config.eval_steps == 0:
                    with Timer(name="val_step", logger=None) as val_step_timer:
                        val_metrics = self.val()
                        metrics_mgr.add_metrics(val_metrics)
                        metrics_mgr.add_metric("time/val_step", val_step_timer.last)

                # 创建一个空的 DataProto 对象，用于后续收集数据。
                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                # === 4. 数据生成（Experience Generation） ===
                # 要按domain group by生成对应的batch
                with actor_infer_timer, actor_infer_response_timer, Timer(
                    name="step_generate", logger=None
                ) as step_generate_timer:
                    domain_batches = {}             # 按 domain（领域）分组存放生成的批次数据
                    batch.meta_info["generation_config"] = self.actor_infer.worker_config.generating_args.to_dict()
                    # 启动 actor_infer 的服务，准备接收 prompt 并生成 response
                    self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
                    # 加载所有奖励模型到GPU，因为马上要用它们来打分
                    for reward_cluster in self.rewards.values():
                        reward_cluster.load_states()

                    batch.meta_info["is_offload_states"] = False
                    # meta mainly for dynamic reward threshold, such as global_step/max_steps
                    batch.meta_info.update(
                        {
                            "global_step": self.global_step,
                            "max_steps": self.pipeline_config.max_steps,
                            "is_training": True,
                        }
                    )
                    
                    # 使用 Ray 进行分布式数据生成
                    scheduler_refs = {}
                    # 遍历每个 domain 的数据调度器 (scheduler)
                    for domain, scheduler in self.generate_schedulers.items():
                        # 异步地调用每个 scheduler 的 get_batch 方法来获取一批 prompts。
                        # .remote() 表示这是一个远程调用，会立即返回一个 future 对象 (ObjectRef)。
                        scheduler_refs[domain] = scheduler.get_batch.remote(
                            data=batch, batch_size=self.domain_batch_size[domain]
                        )
                    # 等待并收集所有 domain 的生成结果
                    for domain, scheduler_ref in scheduler_refs.items():
                        # ray.get() 会阻塞程序，直到远程任务完成并返回结果。
                        domain_batch: DataProto = ray.get(scheduler_ref, timeout=self.pipeline_config.rpc_timeout)
                        # 收集并规约（reduce）生成过程中产生的度量
                        metrics_mgr.add_domain_metrics(
                            domain, reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        )
                        domain_batches[domain] = domain_batch
                    
                    # 将所有 domain 的数据合并成一个大的 batch
                    generate_output = DataProto.concat([domain_batch for domain_batch in domain_batches.values()])
                    generate_output.meta_info.pop("is_offload_states", None)

                    # 卸载奖励模型，释放显存给后续的计算任务
                    for reward_cluster in self.rewards.values():
                        reward_cluster.offload_states()
                    # 停止 actor_infer 服务，并收集生成过程的度量
                    gen_metrics = self.actor_infer.stop_server()
                    metrics_mgr.add_metrics(reduce_metrics(gen_metrics.meta_info.pop("metrics", {})))
                metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

                # 将生成的数据作为当前 step 的主数据 batch
                batch = generate_output
                # mark here to make megatron get_data_input broadcast with non_batch_tensor
                batch.meta_info["_broadcast_non_tensor_batch"]= True

                # === 5. 计算参考模型对数概率 (Reference Log Probs) ===
                # 这是计算 PPO 中 KL 散度惩罚项所必需的。
                with Timer(name="cal_ref_log_probs", logger=None) as cal_ref_log_probs_timer:
                    ref_log_probs = self.reference.compute_log_probs(batch, blocking=True)
                    metrics_mgr.add_reduced_metrics(ref_log_probs.meta_info.pop("metrics", {}))
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    batch = batch.union(ref_log_probs)
                metrics_mgr.add_metric("time/ref_log_probs_values", cal_ref_log_probs_timer.last)

                # === 6. 计算旧策略的对数概率和价值 (Old Log Probs & Values) ===
                # 这些是在进行策略更新之前，当前策略（旧策略）的输出，是 PPO 算法的核心要素。
                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    batch.meta_info["is_offload_states"] = False
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    batch.meta_info["agg_entropy"] = agg_entropy

                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics_mgr.add_reduced_metrics(values.meta_info.pop("metrics", {}))

                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    metrics_mgr.add_reduced_metrics(old_log_probs.meta_info.pop("metrics", {}))
                metrics_mgr.add_metric("time/old_log_probs", cal_old_logpb_timer.last)

                # === 7. 分领域处理奖励和计算优势 (Reward & Advantage Calculation per Domain) ===
                # 为 batch 中的每个样本添加一个唯一的ID，以便后续恢复顺序
                # group by domain to process reward
                batch.batch["prompt_id"] = torch.arange(batch.batch.batch_size[0], device=batch.batch.device)
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                batch_list = []
                for domain, domain_batch in batch_grouped.items():
                    # 1. get sample level mask
                    with Timer(name="get_sample_level_mask", logger=None) as get_sample_level_mask_timer:
                        domain_batch, mask_metrics = get_sample_level_mask(domain_batch, self.pipeline_config)
                        metrics_mgr.add_domain_metrics(domain, mask_metrics)
                    metrics_mgr.add_metric("time/get_sample_level_mask", get_sample_level_mask_timer.last)

                    # 2. process reward
                    with Timer(name="reward_postprocess", logger=None) as reward_postprocess_timer:
                        domain_batch, response_level_metrics = reward_postprocess(
                            domain_batch, self.pipeline_config, self.running
                        )
                        metrics_mgr.add_domain_metrics(domain, response_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/reward_postprocess": reward_postprocess_timer.last})

                    # 3. compute token level rewards
                    with Timer(name="get_token_reward", logger=None) as get_token_reward_timer:
                        domain_batch, token_level_metrics = compute_token_reward(
                            domain_batch, self.pipeline_config, self.kl_ctrl
                        )
                        metrics_mgr.add_domain_metrics(domain, token_level_metrics)
                    metrics_mgr.add_domain_metrics(domain, {"time/get_token_reward": get_token_reward_timer.last})

                    # 4. compute advantage
                    final_response_mask = domain_batch.batch["final_response_mask"].clone()
                    with Timer(name="compute_advantage", logger=None) as compute_advantage_timer:
                        domain_batch = compute_advantage(
                            data=domain_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                            response_mask=final_response_mask,
                        )
                        domain_metrics = reduce_metrics(domain_batch.meta_info.pop("metrics", {}))
                        metrics_mgr.add_domain_metrics(domain, domain_metrics)
                        batch_list.append(domain_batch)
                    metrics_mgr.add_domain_metrics(domain, {"time/compute_advantage": compute_advantage_timer.last})

                # 将处理完的各个 domain 的 batch 重新合并
                batch = DataProto.concat(batch_list)
                # 根据之前保存的 prompt_id 恢复原始顺序
                batch.reorder(indices=torch.argsort(batch.batch["prompt_id"]))
                batch.pop("prompt_id")

                # === 8. 收集并记录所有度量 ===
                metrics_mgr.timers["tps"] = tps_timer
                metrics_mgr.timers["actor_infer"] = actor_infer_timer
                metrics_mgr.timers["actor_infer_response"] = actor_infer_response_timer
                metrics_mgr.timers["actor_train"] = actor_train_timer

                metrics_mgr.add_all_metrics(
                    global_step,
                    batch,
                    resource_manager=self.resource_manager,
                    actor_infer=self.actor_infer,
                    actor_train=self.actor_train,
                )
                batch_grouped: Dict[str, DataProto] = batch.group_by("domain")
                metrics_mgr.add_domain_all_metrics(global_step, batch_grouped)

                # === 9. 执行模型训练步骤 (Train Step) ===
                with Timer(name="step_train", logger=None) as step_train_timer:
                    # 异步地让 critic 模型进行一步训练
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # 实现 critic 预热（warmup）：在训练初期只训练 critic，让价值估计更稳定
                        # implement critic warmup
                        if self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics_mgr.add_reduced_metrics(actor_train_metrics.meta_info.pop("metrics", {}))

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics_mgr.add_reduced_metrics(critic_train_metrics.meta_info.pop("metrics", {}))

                    metrics_mgr.add_metric("time/step_train", step_train_timer.last)

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                actor_infer_response_timer.push_units_processed(
                    n=torch.sum(batch.batch["response_mask"]).detach().item()
                )
                actor_train_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                metrics = metrics_mgr.get_metrics()
                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)
                for domain, scheduler in self.generate_schedulers.items():
                    self.state.kv[f"scheduler_state_{domain}"] = ray.get(scheduler.get_scheduler_state.remote())

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompts = self.tokenizer.batch_decode(generate_output.batch["prompts"], skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(
                        generate_output.batch["responses"], skip_special_tokens=True
                    )
                    generate_examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)][:10]
                    logger.info(json.dumps(generate_examples, ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            batch.meta_info["is_offload_states"] = False
            batch.meta_info["generation_config"] = self.pipeline_config.validation.generating_args.to_dict()
            batch.meta_info.update(
                {"global_step": self.global_step, "max_steps": self.pipeline_config.max_steps, "is_training": False}
            )

            self.actor_infer.start_server(data=DataProto(meta_info=batch.meta_info))
            for reward_cluster in self.rewards.values():
                reward_cluster.load_states()
            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(data=batch, batch_size=len(self.val_dataset)),
                timeout=self.pipeline_config.rpc_timeout,
            )
            self.actor_infer.stop_server()
            generate_output.meta_info.pop("is_offload_states", None)
            for reward_cluster in self.rewards.values():
                reward_cluster.offload_states()
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_score_mean = batch.batch["scores"].detach().float().mean().item()
        val_metrics_mgr.add_metric("val_score/all/mean", val_score_mean)
        logger.info(json.dumps({"val_score/all/mean": val_score_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])

        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            print(f"{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_score", {f"{group_key}/mean": group_batch.batch["scores"].detach().float().mean().item()}
            )

        return val_metrics_mgr.get_metrics()
