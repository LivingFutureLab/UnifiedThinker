# -*- coding: utf-8 -*-
"""
Shop Simulator Environment for ROLL Framework
"""

import random
import threading
import string
import re
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func
from roll.utils.random_utils import all_seed
from roll.utils.logging import get_logger
from .client import MTShopSimulatorClient
import time
import requests

# Thread-local client factory
_tls = threading.local()


def _thread_client(base_url: str, authorization: str, server_key: str, timeout: float, max_retries: int, backoff: float) -> MTShopSimulatorClient:
    """每个线程只创建一个 http client，减少 TCP 握手开销"""
    if not hasattr(_tls, "client"):
        _tls.client = MTShopSimulatorClient(
            base_url=base_url,
            authorization=authorization,
            server_key=server_key,
            timeout=timeout,
            max_retries=max_retries,
            backoff=backoff,
        )
    return _tls.client


class MTShopSimulatorEnv:
    """
    Shop Simulator Environment for ROLL Framework
    Uses HTTP API client to interact with remote environment
    """
    # 添加类级别的锁字典，用于不同文件的锁
    _file_locks = {}
    _locks_lock = threading.Lock()

    def __init__(self,
                 render_mode="text",
                 max_steps=30,
                 format_penalty=-0.05,
                 action_pattern="^<answer>(.*?)</answer>$",
                 special_token_list=("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"),
                 reward_mode="hard",
                 base_url="http://localhost:8000",
                 authorization="Bearer t-ir45wrzaoal7stt1",
                 server_key=None,
                 timeout=10.0,
                 max_retries=3,
                 backoff=1.0,
                 json_dir="/data/oss_bucket_0",
                 mode="train",
                 max_turns=30,
                 shopper_system_template="",
                 shopper_user_template_init="",
                 shopper_model_name="",
                 shopper_base_url="",
                 shopper_authorization="",
                 shopper_xrl_ant="",
                 **kwargs):
        """Initialize ShopSimulatorEnv"""
        # HTTP客户端参数
        self.base_url = base_url
        self.authorization = authorization
        self.server_key = server_key

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        
        # 环境参数
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        
        # 任务控制参数
        self.mode = mode
        self.max_turns = max_turns
        
        # 日志和存储
        self.json_dir = json_dir
        
        # Shopper相关参数
        self.shopper_system_template = shopper_system_template
        self.shopper_user_template_init = shopper_user_template_init
        self.shopper_model_name = shopper_model_name
        self.shopper_base_url = shopper_base_url
        self.shopper_authorization = shopper_authorization
        self.shopper_xrl_ant = shopper_xrl_ant
        
        # Environment instruction - match sokoban pattern
        self.env_instruction = (
            "You are a shopping assistant in a virtual shop simulator. "
            "You need to help customers find products, answer questions, and complete shopping tasks. "
            "You can search for products, click on items, and provide helpful information. "
            "The answer must be one of action in a turn, format is <answer>action</answer>"
        )
        
        # Logging and threading
        self.logger = get_logger()
        self._lock = threading.Lock()
        
        # Runtime state
        self._env_idx: Optional[int] = None
        self._render_cache: str = ""
        self._remaining: int = self.max_steps
        self._history: List[Dict[str, str]] = []
        self._history_shopper: List[Dict[str, str]] = []  # 添加 shopper 历史记录
        self._used_task_ids: set = set()
        
        # Worker identification
        self._worker_id: str = (
            kwargs.get("worker_id")
            or f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
        )
        
        # Step counter - required by traj_env_manager_shop.py
        self.num_env_steps = 0
        
        # Additional attributes for compatibility
        self.room_state = None  # For compatibility with sokoban-like environments
        self.room_fixed = None

    def get_instructions(self) -> str:
        """Get environment instructions for the agent - required by traj_env_manager_shop.py"""
        return self.env_instruction

    def get_task_suffix(self) -> Any:
        """Get task-specific suffix for the current state - required by traj_env_manager_shop.py"""
        if self.render_mode == "text":
            return (
                f"Here is the current state of the shop simulator:\n{self.render(mode='text')}\n"
            )
        else:
            return self.render(mode=self.render_mode)

    def train_task_generator(self):
        """Generate training task IDs"""
        all_task_ids = set(range(1, 1001))
        available_task_ids = all_task_ids - self._used_task_ids
        if not available_task_ids:
            self._used_task_ids.clear()
            available_task_ids = all_task_ids
            self.logger.info("所有任务ID都已使用过，重置已使用任务ID集合")
        
        selected_task_id = random.choice(list(available_task_ids))
        self._used_task_ids.add(selected_task_id)
        return selected_task_id

    def get_task_idx(self):
        train_num = 21962
        eval_num = 1459
        """Get task index based on current mode"""
        if self.mode == "train":
            #return self.train_task_generator()
            return random.randint(eval_num, eval_num + train_num)
        elif self.mode == "val_ind":
            return random.randint(0, eval_num)
        elif self.mode == "val_ood":
            return random.randint(0, eval_num)
        elif self.mode == "val":
            return random.randint(0, eval_num)
        else:
            self.logger.warning(
                f"Unknown mode in ShopSimulatorEnv '{self.mode}', defaulting to 'train'."
            )
            return self.train_task_generator()

    def _force_release(self):
        """Release backend environment resources"""
        if self._env_idx is not None:
            try:
                _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).release_one(self._env_idx)
            except Exception as e:
                self.logger.error(f"release previous env_idx {self._env_idx} failed: {e}")
                pass
            finally:
                self._env_idx = None
        self._render_cache = ""
        self._remaining = self.max_steps
        self._history = []
        self._history_shopper = []  # 重置 shopper 历史记录

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Reset environment for a new episode - required by traj_env_manager_shop.py"""
        for attempt in range(1, 101):
            with self._lock:
                with all_seed(seed):
                    _session = "".join(random.choices(string.ascii_lowercase, k=10))
                    self._force_release()
                    task_idx = kwargs.get("task_idx", self.get_task_idx())
                
                try:
                    rsp = _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).reset(task_idx)
                except Exception as e:
                    self.logger.error(f"Failed to reset environment: {e}")
                    if attempt < 100:
                        import time
                        time.sleep(10)
                        continue
                    else:
                        raise RuntimeError(f"Failed to reset environment after 100 attempts: {e}")

                if "env_idx" in rsp:
                    obs = rsp.get("instruction", "")
                    self._env_idx = rsp["env_idx"]
                    self.task_idx = task_idx
                    self.seed = seed
                    self._remaining = self.max_steps
                    self.num_env_steps = 0

                    instruction = obs.split('Instruction: ')[-1].strip().split('[SEP]')[1]
                    availabel_action = "Env: 搜索功能是否可用" + obs.split("搜索功能是否可用")[-1]
                    self._instruction_fuzzy = "Shopper: " + self.init_shopper(instruction)
                    self._update_render(self._instruction_fuzzy)
                    obj = self._instruction_fuzzy + "\n" + availabel_action
                    self._history = [{"role": "user", "content": obj}]
                    return obj + "\n" + availabel_action, {"suffix": self.get_task_suffix()}
                if "error" in rsp:
                    self.logger.error(f"Failed to reset environment: {rsp['error']}")
                    if attempt < 100:
                        time.sleep(10)
                        continue
                
            if attempt < 20:
                self.logger.warning(
                    f"[ShopSimulatorEnv.reset] backend returned no task idx {task_idx} "
                    f"(attempt {attempt}/100). Retrying in 10 s …"
                )
                import time
                time.sleep(10)

        raise RuntimeError(
            "[ShopSimulatorEnv.reset] backend failed to return env_idx after 100 retries."
        )

    def _get_file_lock(self, file_path):
        """获取文件对应的锁"""
        with self._locks_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]
    
    def _safe_write_log(self, log_file, step_log):
        """安全地写入日志文件，处理多worker并发写入"""
        # 使用线程锁避免同一进程内的并发问题
        file_lock = self._get_file_lock(log_file)
        with file_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_log, ensure_ascii=False) + "\n")
                f.flush()

    def step(self, action: str, global_step: int) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Execute an action in the environment - required by traj_env_manager_shop.py"""
        with self._lock:
            if self._env_idx is None:
                info = {
                    "step_before_reset": 1.0, 
                    "action_is_valid": 0.0, 
                    "reward": 0.0, 
                    "success": 0.0
                }
                return "", 0.0, True, False, info

            # Parse action
            action_info = self.parse_action_mt(action)
            self._history.append({"role": "assistant", "content": action})
            
            if not action_info["action_content"] or action_info["action"] not in ["ask_shopper", "interact_with_env"]:
                reward = self.format_penalty
                reward_new = self.format_penalty
                obs = f"在第{self.num_env_steps}轮, 您返回的action无法解析。请检查格式是否正确。"
                metrics = {
                    'success': 0,
                    'reward_new': reward_new,
                    'reward_old': reward,
                    'r_type': 0,
                    'r_att': 0,
                    'r_option': 0,
                    'r_price': 0
                }
                info = {
                    "suffix": self.get_task_suffix(),
                    "metrics": metrics,
                }
                info.update(action_info)
                done_backend = False
                error_msg = None
                rsp = {}
            else:
                if action_info["action"] == "ask_shopper":
                    # 与shopper交互
                    try:
                        self._history_shopper.append({"role": "user", "content": f"{action_info['action_content']}"})
                        rsp = self.get_response_from_whale_online(messages=self._history_shopper)
                        self._history_shopper.append({"role": "assistant", "content": rsp['response']})
                        obs = "Shopper: " + rsp['response']
                        reward = 0
                        reward_new = 0
                        done_backend = False
                        error_msg = None
                    except Exception as e:
                        self.logger.error(f"Failed to interact with backend: {e}")
                        # Return error state
                        metrics = {
                            'success': 0,
                            'reward_new': 0,
                            'reward_old': 0,
                            'r_type': 0,
                            'r_att': 0,
                            'r_option': 0,
                            'r_price': 0
                        }
                        info = {
                            "suffix": self.get_task_suffix(),
                            "metrics": metrics,
                        }
                        return f"Error: {e}", -1.0, True, False, info
                elif action_info["action"] == "interact_with_env":
                    try:
                        rsp = _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).interact(self._env_idx, action_info["action_content"])
                    except Exception as e:
                        self.logger.error(f"Failed to interact with backend: {e}")
                        # Return error state
                        metrics = {
                            'success': 0,
                            'reward_new': 0,
                            'reward_old': 0,
                            'r_type': 0,
                            'r_att': 0,
                            'r_option': 0,
                            'r_price': 0
                        }
                        info = {
                            "suffix": self.get_task_suffix(),
                            "metrics": metrics,
                        }
                        return f"Error: {e}", -1.0, True, False, info

                    # Update state
                    # 需要隐藏 instruction
                    obs = self.process_observation(rsp.get("instruction", ""))
                    obs = "Env: " + obs
                    reward = float(rsp.get("reward", 0.0))
                    reward_detail =  rsp.get("reward_detail", {})
                    r_type = float(reward_detail.get('r_type', 0.0))
                    r_att = float(reward_detail.get('r_att', 0.0))
                    r_option = float(reward_detail.get('r_option', 0.0))
                    r_price = float(reward_detail.get('r_price', 0.0))
                    reward_new = r_type * r_att * r_option * r_price
                    done_backend = bool(rsp.get("done") or rsp.get("over"))
                    error_msg = rsp.get("error")
                
            self._remaining -= 1
            self.num_env_steps += 1
            self._history.append({"role": "user", "content": obs})
                
            # Determine termination
            terminated = done_backend or bool(error_msg)
            forced_done = self._remaining <= 0
            done = terminated or forced_done
            # Organize info - ensure compatibility with traj_env_manager_shop.py
            backend_info: Dict[str, Any] = rsp.get("info", {}) or {}
            
            # 选择使用哪种reward模式
            if self.reward_mode == "hard":
                info: Dict[str, Any] = {"reward": reward_new}
            else:
                info: Dict[str, Any] = {"reward": reward}

            info["success"] = done_backend
            info["error_flag"] = 0.0
            info["forced_done"] = 0.0

            info['metrics'] = {
                                'success': float(rsp.get('done', 0.0)),
                                'reward_new': reward_new,
                                'reward_old': reward,
                                'r_type': float(reward_detail.get('r_type', 0.0)),
                                'r_att': float(reward_detail.get('r_att', 0.0)),
                                'r_option': float(reward_detail.get('r_option', 0.0)),
                                'r_price': float(reward_detail.get('r_price', 0.0))
                            }

            # 只在episode结束时写入整个交互路径
            if done:
                episode_log = {
                    "global_step": global_step,
                    "seed": self.seed,
                    "task_idx": self.task_idx,
                    "env_idx": self._env_idx,
                    "done": done,
                    "terminated": terminated,
                    "forced_done": forced_done,
                    "total_reward": reward,
                    "total_reward_new": reward_new,
                    "reward_detail": rsp.get("reward_detail", {}),
                    "purchase": rsp.get("purchase", {}),
                    "goal": rsp.get("goal", {}),
                    "conversation": self._history,
                    "total_steps": self.num_env_steps,
                }

                log_dir = os.path.join(self.json_dir, f"global_step_{global_step}")
                log_file = os.path.join(log_dir, f"episode_log_{self._worker_id}.jsonl")

                os.makedirs(log_dir, exist_ok=True)
                self.logger.info(f"Logging episode to {log_file}")
                
                self._safe_write_log(log_file, episode_log)
            
            
            # Add backend metrics
            for k, v in backend_info.items():
                if isinstance(v, (bool, int, float)):
                    info[k] = float(v)

            # Handle errors
            if error_msg:
                info["error_flag"] = 1.0
                self.logger.info(f"error action: {repr(action)}")
                self.logger.info(f"error message: {error_msg}")

            # Handle forced termination
            if forced_done:
                info["forced_done"] = 1.0

            # Resource cleanup
            if done:
                self._force_release()

            self._update_render(obs)
            return obs, reward, terminated, forced_done, info
                
    def process_observation(self, observation):
        # 找到第一个 [SEP]
        first_sep = observation.find("Instruction: [SEP]")

        # 从第一个 [SEP] 之后找第二个 [SEP]
        second_sep = observation.find("[SEP]", first_sep + len("Instruction: [SEP]"))

        # 取第二个 [SEP] 之后的内容
        result = "[SEP]" + observation[second_sep + len("[SEP]"):]
        return result

    def get_response_from_whale_online(self, messages = []):
        # 添加重试机制和更好的错误处理
        retry_delay = 2
        attempt = 0
        data = {"model":self.shopper_model_name, "messages":messages}

        while True:
            try:
                # 设置更详细的请求配置
                session = requests.Session()
                session.headers.update({
                    "Content-Type": "application/json", 
                    "Authorization": self.shopper_authorization, 
                    "XRL-Authorization": self.shopper_xrl_ant,
                    "BusinessType": "online"
                })
                
                # 设置超时和重试
                response = session.post(
                    self.shopper_base_url,
                    json=data,
                    timeout=10,  # (连接超时, 读取超时)
                    verify=True  # 验证SSL证书
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    content  =  response_json['choices'][0]['message']['content']
                    # return: env_idx, idx, instruction(ins+obs), message("继续交互"), over(bool), reward(float), error_msg(str), done(bool)
                    rsp = {'response': content, 'message': ''}
                    return rsp
                else:
                    print(f'HTTP错误: {response.status_code}')
                    print(f'响应内容: {response.text}')
            except requests.exceptions.ConnectTimeout as e:
                print(f"连接超时错误 (尝试 {attempt + 1}): {e}")
            except requests.exceptions.ReadTimeout as e:
                print(f"读取超时错误 (尝试 {attempt + 1}): {e}")
            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 (尝试 {attempt + 1}): {e}")
            except requests.exceptions.RequestException as e:
                print(f"请求错误 (尝试 {attempt + 1}): {e}")
            except Exception as e:
                print(f"未知错误 (尝试 {attempt + 1}): {e}")
            print(f"等待 {retry_delay} 秒后重试...")
            if attempt % 50 == 0: 
                self.logger.error(f"[whale推理]重试{attempt}次都失败了, 重试等待时间{retry_delay+attempt//10}秒....")
            time.sleep(retry_delay+attempt//10)
            attempt += 1

    def parse_action(self, text: str) -> Dict[str, Any]:
        """Parse action text using the default parser - required by traj_env_manager_shop.py"""
        
        return default_parser_action_func(
            text, 
            self.action_pattern, 
            {},  # No specific action lookup for shop simulator
            self.special_token_list
        )

    def parse_action_mt(self, text: str) -> Dict[str, Any]:
        
        # 匹配新的格式
        pattern = self.action_pattern
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            thought = match.group(1).strip() if match.group(1) else ""
            action_type = match.group(2).strip() if match.group(2) else ""
            action_content = match.group(3).strip() if match.group(3) else ""
            
            return {
                "action_content": action_content,
                "action": action_type,
                "think_content": thought,
                }
        else:
            return {
                "action_content": "",
                "action": "",
                "think_content": ""
            }

    def init_shopper(self, instruction):
        self._history_shopper = [
                                {"role": "system", "content": self.shopper_system_template.format(goal=instruction)},
                                {"role": "user", "content": self.shopper_user_template_init}
                            ]
        rsp = self.get_response_from_whale_online(messages=self._history_shopper)
        self._history_shopper.append({"role": "assistant", "content": rsp['response']})
        return rsp['response']

    def render(self, mode: str = "text") -> Union[str, np.ndarray]:
        """Render the current environment state - required by traj_env_manager_shop.py"""
        if mode == "rgb_array":
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return self._render_cache

    def close(self):
        """Manually release backend resources"""
        with self._lock:
            self._force_release()

    def release_lst(self, env_lst: List[int], release_all: bool = False):
        """Release multiple environments"""
        if release_all:
            _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).release_all()
        else:
            for env_idx in env_lst:
                _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).release_one(env_idx)

    def release_one(self, env_idx: int):
        """Release a single environment"""
        _thread_client(self.base_url, self.authorization, self.server_key, self.timeout, self.max_retries, self.backoff).release_one(env_idx)

    def sample_random_action(self):
        """Sample a random action (for testing purposes)"""
        actions = ["search[product]", "click[item]", "help", "navigate"]
        return random.choice(actions)

    def _update_render(self, obs: str):
        """Update the render cache with current observation"""
        action_pattern = r"search\[.*?\]|click\[.*?\]|help|navigate"
        acts = re.findall(action_pattern, obs, re.IGNORECASE)
        
        if acts:
            self._render_cache = obs + "\nAvailable actions: " + ", ".join(acts)
        else:
            self._render_cache = obs

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self._force_release()
        except Exception:
            pass



if __name__ == "__main__":
    # Test the environment
    env = MTShopSimulatorEnv()
    
    # Test reset
    obs, info = env.reset(seed=42)
    print("Reset observation:", obs)
    print("Info:", info)
    
    # Test step
    action = "<answer>search[shoes]</answer>"
    obs, reward, terminated, truncated, info = env.step(action, global_step=0)
    print("Step result:", obs, reward, terminated, truncated)
    
    # Test render
    rendered = env.render()
    print("Rendered:", rendered)
    
    env.close() 