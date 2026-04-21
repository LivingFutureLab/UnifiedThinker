import asyncio
import json
import logging
import os
import random
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import datasets
import numpy as np
import ray
from datasets import Dataset, DatasetDict, load_dataset
from gem import Env

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.agentic.env.rock.sanbox_manager import SandboxManager
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.random_utils import all_seed

logger = logging.getLogger(__name__)


class TerminalBenchEnv(Env):
    """Terminal-bench environment for ROLL"""
    def __init__(
        self,
        group_id: int = 0,
        num_env_groups: int = 1,
        max_steps: int = 80,
        mode: str = "train",
        xrl_authorization: str = "",
        sandbox_base_url: str = "https://xrl.alibaba-inc.com",
        user_id: str = "0000",
        experiment_id: str = "test",
        auto_clear_seconds: int = 60 * 60, 
        agent_timeout_sec: int = 60 * 40,
        test_timeout_sec: int = 60 * 40,
        max_env_time: float = 60 * 50,
        run_type: str = "iflow-cli",
        iflow_base_url: str = "https://apis.iflow.cn/v1",
        iflow_selected_auth_type: str = "iflow",
        search_api_key: str = "",
        iflow_key: str = "",
        debug: bool = False,
        test_files: List[str] = None,
        agent_version: str = "0.2.26",
        dataset_name: Optional[str] = "",
        train_idx_range: Tuple[int, int] = (0, 1e9),  # 训练集任务ID范围
        val_idx_range: Tuple[int, int] = (0, 1e9),  # 验证集任务ID范围
        pass_low_threshold: float = 0.0,
        pass_high_threshold: float = 0.0,
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        id_key: str = "id",
        question_key: str = "prompt",
        sandbox_image_key: str = "sandbox_image",
        task_name_key: str = "task_name",
        run_region_key: str = "run_region",
        seed: int = 0,
        group_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Terminal-bench environment"""
        super().__init__()
        self.mode = mode
        self.debug = debug
        self.train_idx_range = train_idx_range
        self.val_idx_range = val_idx_range
        self.pass_low_threshold = pass_low_threshold
        self.pass_high_threshold = pass_high_threshold
        
        # 设置环境标识
        self.group_id = group_id
        self.num_env_groups = num_env_groups
        self.seed = seed
        self.group_key = group_key
        
        # 设置超时参数
        self.agent_timeout_sec = agent_timeout_sec
        self.test_timeout_sec = test_timeout_sec
        self.auto_clear_seconds = auto_clear_seconds
        self.max_env_time = max_env_time
        
        self.time_start = 0
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.traj_rollout_time = 0
        
        self.env_timeout = False
        self.env_reset_failed = False
        
        # 设置任务和数据集相关参数
        self.id_key = id_key
        self.question_key = question_key
        self.sandbox_image_key = sandbox_image_key
        self.task_name_key = task_name_key
        self.run_region_key = run_region_key
        self.test_files = test_files
        self.task_id = -1
        self.task_name = ""
        self.prompt = ""
        self.run_region = ""
        
        # xrl相关参数
        self.xrl_authorization = xrl_authorization
        self.sandbox_base_url = sandbox_base_url
        self.run_type = run_type
        self.iflow_base_url = iflow_base_url
        self.iflow_selected_auth_type = iflow_selected_auth_type
        self.search_api_key = search_api_key
        self.iflow_key = iflow_key
        self.agent_version = agent_version
        self.user_id = user_id
        self.experiment_id = experiment_id
        # 数据集读取
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                             mode=global_dataset_mode)
        # 这里开启做数据过滤，待进一步补充更多
        # TODO 不同环境的基本字段对齐
        # 通过index保留特定条数   
        data_ranges = self.val_idx_range if self.mode == "val" else self.train_idx_range
        ray.get(
            self.dataset.filter.remote(
                filter_name="filter_idx_range", function=lambda x: data_ranges[0] <= int(x["id"]) <= data_ranges[1]
            )
        ) 
        ray.get(
            self.dataset.filter.remote(
                filter_name="filter_pass_range", function=lambda x: self.pass_low_threshold <= int(x["pass_ratio"]) <= self.pass_high_threshold
            )
        ) 
   
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                             get_if_exists=True,
                                                             namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        
        # 设置环境参数
        self.max_steps = max_steps
        self.is_initialized = False
        self.terminated = False
        self.truncated = False
        self.current_step = 0
        
        # 失败模式和停止原因
        self.failure_mode = ""
        self.stop_reason = ""
        self.error_messages = []
        
        # 测试结果
        self.test_output = ""
        
        self.logger = get_logger()
        
        # self.reset()

    def step(
        self, response: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step in the environment"""
        self.logger.info(f"[ENV_STEP] START - GroupID: {self.group_id}, Step:{self.current_step + 1}, Task:{self.task_name}, Response:{json.dumps(response, ensure_ascii=False)}")
        info = {}
        st = time.time()
        self.current_step += 1
        self.rollout_time = round(time.time() - self.time_start, 4)
        if self.rollout_time > self.max_env_time:
            self.traj_total_time = round(self.traj_reset_time + self.traj_step_time, 4)
            self.env_timeout = True
            
            metrics = {
                "env_timeout": True,
                "env_reset_failed": False,
                "action_is_valid": 0,
                "success": False,
                "raw_reward": 0,
                "traj_reset_time": self.traj_reset_time,
                "traj_step_time": self.traj_step_time,
                "traj_total_time": self.traj_total_time,
                "traj_rollout_time": self.traj_rollout_time,
                "current_step": self.current_step,
                "task_id": self.task_id
            }
            
            metrics_agg_mode = {
                "action_is_valid": "mean",
                "success": "last",
                "raw_reward": "last",
            }
            
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "error": "ENV_TIMEOUT"
            }
            
            observation = f"ERROR: 环境交互时间超过{self.max_env_time/60}分钟限制"
            self.logger.error(f"[ENV_STEP] Failed! - Environment timeout after {self.max_env_time/60} minutes")
            # if not self.debug:
            #     self.close()
            return observation, 0.0, True, False, info
        
        try:
            observation, reward, terminated, truncated, info = self.sandbox_util.process_model_response(
                response, self.agent_timeout_sec, self.task_id, self.task_name
            )
            
            sandbox_failure_mode = getattr(self.sandbox_util, 'failure_mode', '')
            self.error_messages.extend(getattr(self.sandbox_util, 'error_messages', []))
            
            if sandbox_failure_mode and sandbox_failure_mode != 'none':
                self.failure_mode = sandbox_failure_mode
            
            if not terminated and self.current_step >= self.max_steps:
                terminated = True
                self.logger.info(f"[MAX_STEPS] Reached maximum steps ({self.max_steps}), truncating episode, task_name: {self.task_name}")
                observation = f"ERROR: Reached maximum steps {self.max_steps}"
                self.stop_reason = "Reached maximum steps"
                
            if terminated:
                st_reward = time.time()
                self.logger.info(f"[REWARD_CALC] START - Calculating reward for task_id: {self.task_id}, task_name: {self.task_name}")
                reward, error_info, test_output = self.calculate_reward()
                self.test_output = test_output
                if error_info:
                    self.logger.error(f"[REWARD_CALC] Failed! - Error during reward calculation: {error_info}")
                    if error_info and error_info != '' and (not self.failure_mode or self.failure_mode == ''):
                        self.failure_mode = "reward_calculation_failed"
                        self.error_messages.append(error_info)
                else:
                    self.logger.info(f"[REWARD_CALC] Success! - Final reward: {reward} for task_id: {self.task_id}, task_name: {self.task_name}")
                self.traj_reward_time = round(time.time() - st_reward, 4)
                if not self.debug:
                    self.sandbox_util.stop_sandbox()
        
            self.traj_step_time += round(time.time() - st, 4) - getattr(self, 'traj_reward_time', 0)
            self.traj_total_time = round(self.traj_reset_time + self.traj_step_time + getattr(self, 'traj_reward_time', 0), 4)
            self.traj_rollout_time = round(time.time() - self.time_start, 4)
            
            action_is_valid = info.get("action_is_valid", 0)
            metrics = {
                "env_timeout": self.env_timeout,
                "env_reset_failed": self.env_reset_failed,
                "action_is_valid": action_is_valid,
                "success": reward > 0,
                "raw_reward": reward,
                "traj_reset_time": self.traj_reset_time,
                "traj_step_time": self.traj_step_time,
                "traj_reward_time": getattr(self, 'traj_reward_time', 0),
                "traj_total_time": self.traj_total_time,
                "traj_rollout_time": self.traj_rollout_time,
                "current_step": self.current_step,
                "task_id": self.task_id
            }
            
            if self.current_step >= self.max_steps:
                terminated = True
            
            metrics_agg_mode = {
                "action_is_valid": "mean",
                "success": "last",
                "raw_reward": "last",
            }
            info_new = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "failure_mode": self.failure_mode,
                "stop_reason": self.stop_reason,
                "error_messages": self.error_messages,
                "test_output": self.test_output
            }
            info.update(info_new)
            self.logger.info(f"[ENV_STEP] Success! - Step {self.current_step} finished, reward: {reward}, success: {reward > 0}")
            return observation, reward, terminated, truncated, info
                
        except Exception as e:
            self.logger.error(f"[ENV_STEP] Failed! - Error processing model response: {e}")
            info = {
                "failure_mode": self.failure_mode,
                "stop_reason": self.stop_reason,
                "error_messages": self.error_messages
            }
            return f"Error processing response: {str(e)}", 0.0, terminated, truncated, info

    def get_sysinfo(self, prompt: str = "") -> Dict[str, Any]:
        """Get system information"""
        self.logger.info(f"[GET_SYSINFO] START - Getting system info for prompt: {(prompt or self.prompt)[:100]}...")
        try:
            if prompt:
                messages, error_info = self.sandbox_util.get_messages(prompt)
            else:
                messages, error_info = self.sandbox_util.get_messages(self.prompt)
            
            if error_info:
                self.logger.error(f"[GET_SYSINFO] Failed! - Error: {error_info}")
                self.error_messages.append(f"GET_SYSINFO_ERROR: {error_info}")
            else:
                self.logger.info(f"[GET_SYSINFO] Success! - Retrieved {len(messages)} messages")
            
            return messages, error_info
        except Exception as e:
            self.logger.error(f"[GET_SYSINFO] Failed! - Exception: {str(e)}")
            self.error_messages.append(f"GET_SYSINFO_EXCEPTION: {str(e)}")
            return [], str(e)


    def calculate_reward(self) -> float:
        """Calculate the reward for the current episode"""
        reward = 0
        is_resolved, error_info, test_output = self.sandbox_util.run_tests(
            self.test_files, 
            self.test_timeout_sec, 
            self.task_name
        )
        if is_resolved:
            reward = 1
        return reward, error_info, test_output

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and start a new episode"""
        self.logger.info(f"[ENV_RESET] START - Resetting environment with seed: {seed}")
        
        try:
            super().reset(seed)
            
            self.env_timeout = False
            self.env_reset_failed = False
            self.terminated = False
            self.truncated = False
            self.current_step = 0
            
            self.traj_reset_time = 0
            self.traj_step_time = 0
            self.traj_reward_time = 0
            self.traj_total_time = 0
            self.traj_rollout_time = 0
            
            self.failure_mode = ""
            self.stop_reason = ""
            self.error_messages = []
            self.test_output = ""
            task_id, data = self.get_task_idx_and_data(seed)
            if data is None:
                return None, {}
            
            self.task_id = data[self.id_key]
            self.prompt = data[self.question_key]
            self.run_region = data[self.run_region_key]
            self.sandbox_image = data[self.sandbox_image_key]
            self.task_name = data[self.task_name_key]
            
            # 拉起沙盒服务
            self.start_sandbox()
            return self.prompt, {}
            
        except Exception as e:
            self.env_reset_failed = True
            self.logger.error(f"[ENV_RESET] Failed! - Error: {str(e)}")
            raise
        
    def start_sandbox(self):
        # 初始化xrl服务和拉起基础镜像
        st0 = time.time()
        self.time_start = st0
        
        self.logger.info(f"[SANDBOX_INIT] START - Initializing sandbox with image: {self.sandbox_image} for task: {self.task_name}")
        st1 = time.time()
        self.sandbox_util = SandboxManager(
            sandbox_image=self.sandbox_image,
            logger=self.logger,
            # xrl相关的参数
            xrl_authorization=self.xrl_authorization,
            sandbox_base_url=self.sandbox_base_url,
            user_id = self.user_id,
            experiment_id = self.experiment_id,
            # iflow相关的参数
            run_type=self.run_type,
            iflow_base_url=self.iflow_base_url,
            iflow_api_key=self.iflow_key,
            iflow_search_api_key=self.search_api_key,
            iflow_selected_auth_type=self.iflow_selected_auth_type,
            agent_version=self.agent_version,
            run_region=self.run_region,
            debug=self.debug
        )
        self.tools = self.sandbox_util.iflow_tool.tools
        self.traj_reset_time = round(time.time() - st0, 4)

        if not self.sandbox_util.is_environment_available:
            self.env_reset_failed = True
        else:
            self.env_reset_failed = False
            self.logger.info(f"[ENV_RESET] Success! - TaskID: {self.task_id}, TaskName: {self.task_name}, image: {self.sandbox_image}, sandbox_ip: {self.sandbox_util.sandbox_ip}, sandbox_id: {self.sandbox_util.sandbox_id}, Reset time: {self.traj_reset_time}s")
        
    
    def close(self):
        """Close the environment and log final metrics"""
        self.logger.info(f"[ENV_CLOSE] START - Closing environment for task: {getattr(self, 'task_name', 'N/A')}")
        
        try:
            if hasattr(self, 'time_start') and self.time_start > 0:
                self.traj_rollout_time = round(time.time() - self.time_start, 4)
                self.traj_total_time = round(
                    getattr(self, 'traj_reset_time', 0) + 
                    getattr(self, 'traj_step_time', 0) + 
                    getattr(self, 'traj_reward_time', 0), 4
                )
            
            if self.env_timeout:
                self.stop_reason = "ENV_TIMEOUT"
            elif self.env_reset_failed:
                self.stop_reason = "ENV_RESET_FAILED"
            elif self.truncated:
                self.stop_reason = "TRUNCATED"
            elif self.terminated:
                self.stop_reason = "TERMINATED"
            else:
                self.stop_reason = "UNKNOWN"
            
            if self.sandbox_util:
                self.sandbox_util.stop_sandbox()
                self.sandbox_util = None
            
            self.logger.info(f"[ENV_CLOSE] Success! - Environment closed, stop_reason: {self.stop_reason}, total_time: {getattr(self, 'traj_total_time', 0)}s")
                
        except Exception as e:
            self.logger.error(f"[ENV_CLOSE] Failed! - Error during environment close: {e}")
            
    def _filter_dataset_by_group(self, dataset: Dataset) -> Dataset:
        if self.group_key and self.group_key in dataset.features:
            return dataset.filter(lambda example: example[self.group_key] == self.group_id)
        
        if self.num_env_groups > 1:
            return dataset.filter(lambda example, idx: idx % self.num_env_groups == self.group_id, with_indices=True)
        return dataset
    
    
    def get_task_idx_and_data(self,seed):
        data_item: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data_item is None:
            return None, None
        idx = data_item["id"]
        return idx, data_item