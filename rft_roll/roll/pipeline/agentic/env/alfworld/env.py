# pylint: disable=too-many-instance-attributes,line-too-long
import json
import os
import random
import re
import threading
import time
import uuid
from typing import Any, Tuple

import gym
import jsonlines
import yaml
from gem import Env

from roll.pipeline.agentic.env.alfworld.src.agents.environment import get_environment
from roll.pipeline.agentic.env.alfworld.utils import collect_game_files
from roll.utils.random_utils import all_seed
from roll.utils.logging import get_logger

logger = get_logger()

def write_data_json(data,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w',encoding="utf-8") as f:
        info_str = json.dumps(data, ensure_ascii=False)
        f.write(info_str)

class AlfworldGemEnv(Env, gym.Env):
    """
    基于gem的ALFWorld环境，参考frozen环境的实现模式
    """
    # _env_factory_lock = threading.Lock()
    # _base_env_instance = None    
    _tatsu_parser_lock = threading.Lock()
    
    def __init__(self,
                 mode: str = "train",
                 max_turns: int = 30,
                 json_dir: str = "/data/oss_bucket_0/hanyi/RL/Debug",
                 data_yaml:str = "./eval_configs/base_config.yaml",
                 label_path:str = "./data",
                 timeout: float = 10.0,
                 train_idx_range: Tuple[int, int] = (0, 1465),
                 val_ood_idx_range: Tuple[int, int] = (0, 133),
                 backoff: float = 1.0,
                 render_mode: str = "text",
                 format_penalty: float = -0.1,
                 action_pattern: str = "^<answer>(.*?)</answer>$",
                 special_token_list: tuple = ("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"),
                 **kwargs):


        self.data_yaml = data_yaml
        with open(self.data_yaml, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        # 环境设置
        self.render_mode = render_mode
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.json_dir = json_dir
        self.data_yaml = data_yaml
        self.train_idx_range = train_idx_range
        self.val_ood_idx_range = val_ood_idx_range
        self.mode = mode
        self.label_path = label_path
        self.max_turns = max_turns
        # ALFWorld特定的动作空间和指令
        self.env_instruction = (
            "The answer must be one action, format is <think>your thought</think>\n<answer>your action</answer>."
        )
        
        # Worker ID用于日志
        self._worker_id: str = (
            kwargs.get("worker_id")
            or f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
        )

        # 时间记录
        self.time_start = 0 # 起始时间
        self.traj_total_time = 0 # 服务交互时间
        self.traj_rollout_time = 0 # 总rollout时间
        self.traj_step_time = 0 # 服务交互时间
        self.traj_reward_time = 0 # 奖励计算时间
        self.traj_reset_time = 0 # 重置时间

        self.action_is_valid_lst = []
        self.action_is_effective_lst = []
    
    def get_task_id(self, seed):
        print(f'self.train_idx_range: {self.train_idx_range}')
        with all_seed(seed):
            if self.mode == "train":
                task_id = random.randint(self.train_idx_range[0], self.train_idx_range[1])
            elif self.mode == "val_ood":
                task_id = random.randint(self.val_ood_idx_range[0], self.val_ood_idx_range[1])
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        self.labeled_data = []
        self.syf_label={}
        
        if self.mode == "train":
            cur_data_path = self.config["dataset"]["data_path"] + '/train_batch_' + str(task_id//100 + 1) + '/'
            
            label_paths = os.path.join(self.label_path, "rl_train.jsonl")
            
            with open(label_paths, 'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    self.labeled_data.append(cur_data_path + item["additional_info"]['description'])

            with open(label_paths, 'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    self.syf_label[item["additional_info"]['description']] = item

            logger.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]load_data from {cur_data_path}. (cur_task_idx: {task_id})')
            game_files = collect_game_files(self.mode, self.data_yaml, cur_data_path, self.labeled_data)
            print(f"task_id: {task_id}")
            return game_files[task_id%100] 
        elif self.mode == "val_ood":
            cur_data_path = self.config["dataset"]["eval_ood_data_path"]+'/'
            label_paths = os.path.join(self.label_path, "test.jsonl")
            
            with open(label_paths, 'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    self.labeled_data.append(cur_data_path + item["additional_info"]['description'])
            
            with open(label_paths, 'r', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    self.syf_label[item["additional_info"]['description']] = item
            logger.info(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]load_data from {cur_data_path}. (cur_task_idx: {task_id})')
            game_files = collect_game_files(self.mode, self.data_yaml, cur_data_path, self.labeled_data)
            print(len(game_files))
            print(f"task_id: {task_id}")
            return game_files[task_id] 
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
  
        
    
    # def get_instructions(self) -> str:
    #     """返回环境的指令说明"""
    #     return self.env_instruction
    
    def get_task_suffix(self) -> Any:
        """返回当前任务状态的描述"""
        render_cache = self.render(mode="text")
        obs = render_cache.get("obs", "")
        goal = render_cache.get("goal", "")
        step_count = render_cache.get("step_count", 0)
        return (
            f"Here is the current state of ALFWorld:\n"
            f"Current observation: {obs}\n"
            f"Goal: {goal}\n"
            f"Steps taken: {step_count}/{self.max_turns}\n"
        )

    def reset(self, seed=None):
        """重置环境并返回初始观察"""
        st = time.time()
        self.time_start = st
        self.clean_record()
        try:
            with self._tatsu_parser_lock:  
                with all_seed(seed):
                    env = get_environment(self.config["env"]["type"])(self.config, train_eval=self.mode, input_game_files=self.get_task_id(seed))
                    self.env = env.init_env(1)
                    ob, info =self.env.reset()

            self.valid_actions = info["admissible_commands"][0]
            self.init_obs = ('\n'.join(ob[0].split('\n\n')[1:])).split('\n')[0]
            self.goal = (('\n'.join(ob[0].split('\n\n')[1:])).split('\n')[1]).split("Your task is to:")[1].strip()
            self.env_ob = self.init_obs+ "\nThe task goal to be completed is: " + self.goal

            self.cur_task_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            # print(self.cur_task_name)
            
            self.sub_goal = self.syf_label[self.cur_task_name]["subgoals"]
            
            self.difficulty = self.syf_label[self.cur_task_name]["difficulty"]
            self.finished_sub_goal = [0 for i in range(len(self.sub_goal) + 1)]
            

            # 返回指令和任务描述
            # instructions = self.get_instructions()
            self.history = [{'role': 'user', 'content': self.init_obs + self.goal}] 
            # record time
            self.traj_reset_time = round(time.time() - st, 4) 
            self.render_cache = {"goal":self.goal, "observation":self.init_obs, "valid_action_set": self.valid_actions, "env_instruction": self.env_instruction}
            return  self.init_obs, {"suffix": self.render_cache}
            
        except (RuntimeError, RuntimeWarning) as e:
            logger.info(f"Reset failed: {e}")
            # 尝试使用不同的种子重试
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)
    
    def fine_step(self, action: str):
        """执行一步动作"""
        self.step_count += 1
        st = time.time()
        print('[DEBUG][模型输入action]', action)

        # 添加环境检查
        if self.env is None:
            logger.error("Environment is None, attempting to reset")
            return None, -2, False, False, None
        
        truncated = self.step_count >= self.max_turns
        

        action_info = self.parse_action(action)
        print(f'\n[DEBUG][action_info]{action_info}')
        format_info = self.format_exam(action_info)
        print(f'\n[DEBUG][format_info]{format_info}')
        
        self.history.append({"role": "assistant", "content": f'{action}'})
        with self._tatsu_parser_lock: 
            if format_info['error']:
                # 无效动作处理
                terminate_obs = f"{format_info['error_msg']}"
            
                metrics = {
                    "action_is_valid": False,
                    "success": self.isdone,
                    "format_penalty": self.format_penalty
                }
                info = {
                    "suffix": self.get_task_suffix(),
                    "metrics": metrics,
                }
                info.update(action_info)
                self.history.append({"role": "user", "content": f'{terminate_obs}'})

                self.env.close()
                return terminate_obs, -0.1, True, truncated, info
 
            ob, reward, done, env_info = self.env.step([action_info["action"]])
            self.isdone = done[0]
            observation = self._process_ob(ob[0])
            next_obs = observation
            self._check_temperature_string(s=observation, selected_obs=self.sub_goal)

            terminated = self.isdone or truncated

            st_reward = time.time()
            self.traj_reward_time = round(time.time() - st_reward, 4)

            if terminated or truncated:
                self.traj_total_time = self.traj_reset_time + self.traj_step_time + self.traj_reward_time 
            
            #reward计算
            if self.isdone:
                self.reward=10.0
            else:
                self.reward=0.0
            
            self.traj_step_time += round(time.time() - st, 4) - self.traj_reward_time    

            metrics = {
                        "action_is_valid": True,
                        "success": self.isdone,
                        "format_penalty": 0,
                        "reward": self.reward,
                        "traj_reset_time": self.traj_reset_time,
                        "traj_step_time": self.traj_step_time,
                        "traj_reward_time": self.traj_reward_time,
                        "traj_total_time": self.traj_total_time,
                        "step_count": self.step_count,
                    }
            info = {
                "suffix": next_obs,
                "metrics": metrics,
            }        
            info.update(action_info)
            self.history.append({"role": "user", "content": f'{next_obs}'})
            print('-------------------------------------------------')
            print(f'[DEBUG][obs]{next_obs}')
            print(f'[DEBUG][reward]{self.reward}')
            print(f'[DEBUG][terminate]{terminated}')
            print(f'[DEBUG][truncated]{truncated}')
            print(f'[DEBUG][metrics]\n{metrics}')
            print(f'-------------- AlfworldEnv.step(end) -----------------')
            if terminated or truncated:
                stop_reason = "terminate" if terminated else "truncated"
                save = {
                    "task_idx": self.task_idx,
                    "reward": self.reward,
                    "terminate": terminated,
                    "truncated": truncated,
                    "stop_reason": stop_reason,
                    "history": self.history,
                    "metrics": metrics,
                }
                os.makedirs(self.json_dir,exist_ok=True)
                log_path = os.path.join(self.json_dir,f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-{self.task_idx}_re{self.reward}_{stop_reason}_step{self.step_count}_time{self.traj_total_time}_{self._worker_id}.json')
                write_data_json(save,log_path)
                self.env.close()  
                     
            return next_obs, self.reward, terminated, truncated, info


    def step(self, action: str):
        """执行一步动作"""
        self.step_count += 1
        st = time.time()
        print('[DEBUG][模型输入action]', action)
      
        # 添加环境检查
        if self.env is None:
            logger.error("Environment is None, attempting to reset")
            return None, -2, False, False, None

        with self._tatsu_parser_lock: 
            truncated = self.step_count >= self.max_turns

            action_info = self.parse_action(action)
            print(f'\n[DEBUG][action_info]{action_info}')
            self.history.append({"role": "assistant", "content": f'{action}'})
            format_info = self.format_exam(action_info)
            print(f'\n[DEBUG][format_info]{format_info}')
        
            if format_info['error']:
                self.action_is_valid_lst.append(0)
                self.action_is_effective_lst.append(0)
                # 无效动作处理
                terminate_obs = f"{format_info['error_msg']}"
                # reward = -0.1
                metrics = {
                    "action_is_valid": False,
                    "success": self.isdone,
                    "format_penalty": self.format_penalty,
                    "traj_reset_time": self.traj_reset_time,
                    "traj_step_time": self.traj_step_time,
                    "traj_reward_time": self.traj_reward_time,
                    "traj_total_time": self.traj_total_time,
                    "traj_rollout_time": self.traj_rollout_time,
                    "action_is_valid": round(sum(self.action_is_valid_lst)/len(self.action_is_valid_lst), 4) if self.action_is_valid_lst else 0.0,
                    "action_is_effective": round(sum(self.action_is_effective_lst)/len(self.action_is_effective_lst), 4) if self.action_is_effective_lst else 0.0,
                }
                info = {
                    "suffix": self.get_task_suffix(),
                    "metrics": metrics,
                }
                info.update(action_info)
                self.history.append({"role": "user", "content": f'{terminate_obs}'})
                self.env.close()

                return terminate_obs, -1.0, True, truncated, info
            
            ob, reward, done, env_info = self.env.step([action_info["action"]])
            self.isdone = done[0]
            observation = self._process_ob(ob[0])
            next_obs = observation
            
            self._check_temperature_string(s=observation, selected_obs=self.sub_goal)
            
            terminated = self.isdone or truncated
            st_reward = time.time()
            self.traj_reward_time = round(time.time() - st_reward, 4)

            if terminated or truncated:
                self.traj_total_time = self.traj_reset_time + self.traj_step_time + self.traj_reward_time 
            
            #reward计算
            if self.isdone:
                self.reward=1.0
            elif truncated and not self.isdone:
                self.reward=-1.0
            else:
                self.reward=0.0

            if "nothing happens" in observation:
                self.action_is_valid_lst.append(1)
                self.action_is_effective_lst.append(0)
            else:
                self.action_is_valid_lst.append(1)
                self.action_is_effective_lst.append(1)
            

            self.traj_step_time += round(time.time() - st, 4) - self.traj_reward_time    
            self.traj_rollout_time = round(time.time() - self.time_start, 4)

            metrics = {
                        "action_is_valid": True,
                        "success": self.isdone,
                        "format_penalty": 0,
                        "reward": self.reward,
                        "traj_reset_time": self.traj_reset_time,
                        "traj_step_time": self.traj_step_time,
                        "traj_reward_time": self.traj_reward_time,
                        "traj_total_time": self.traj_total_time,
                        "step_count": self.step_count,
                        "traj_rollout_time": self.traj_rollout_time,
                        "action_is_valid": round(sum(self.action_is_valid_lst)/len(self.action_is_valid_lst), 4) if self.action_is_valid_lst else 0.0,
                        "action_is_effective": round(sum(self.action_is_effective_lst)/len(self.action_is_effective_lst), 4) if self.action_is_effective_lst else 0.0,
                    }
            info = {
                "suffix": next_obs,
                "metrics": metrics,
            }        
            info.update(action_info)
            self.history.append({"role": "user", "content": f'{next_obs}'})
            print('-------------------------------------------------')
            print(f'[DEBUG][obs]{next_obs}')
            print(f'[DEBUG][reward]{self.reward}')
            print(f'[DEBUG][terminate]{terminated}')
            print(f'[DEBUG][truncated]{truncated}')
            print(f'[DEBUG][metrics]\n{metrics}')
            print(f'-------------- AlfworldEnv.step(end) -----------------')
            if terminated or truncated:
            #     stop_reason = "terminate" if terminated else "truncated"
            #     save = {
            #         "task_idx": self.task_idx,
            #         "reward": self.reward,
            #         "terminate": terminated,
            #         "truncated": truncated,
            #         "stop_reason": stop_reason,
            #         "history": self.history,
            #         "metrics": metrics,
            #     }
                # os.makedirs(self.json_dir,exist_ok=True)
                # log_path = os.path.join(self.json_dir,f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-{self.task_idx}_re{self.reward}_{stop_reason}_step{self.step_count}_time{self.traj_total_time}_{self._worker_id}.json')
                # write_data_json(save,log_path)
                self.env.close()
                    
            return next_obs, self.reward, terminated, truncated, info
            
    def parse_action(self, text):
        """解析动作文本"""
        action = None
        match = re.search(self.action_pattern, text, re.DOTALL)
        if not match:
            action_info = {
                "response":text,
                "action": action,
                "action_content": "Format Invalid",
                "think_content": "Format Invalid"
            }
            return action_info
        if len(match.groups()) == 1:
            think_content, action_content = "", match.group(1).strip()
        else:
            think_content, action_content = match.group(1).strip(), match.group(2).strip()
        action_content = action_content.strip()
        think_content = think_content.strip()

        action = action_content
        action = re.sub(r'(?<!in/)(?<!on/)\b(in|on)\b', 'in/on', action, flags=re.IGNORECASE)

        action_info = {
            "response":text,
            "action": action,
            "action_content": action_content,
            "think_content": think_content,
        }
        return action_info

    def _check_temperature_string(self, s, selected_obs):
        for i, pattern in enumerate(selected_obs):
            #if self.finished_sub_goal[i] == 1.:
            #    continue
            match = re.search(pattern, s)
            if match:
                self.finished_sub_goal[i] = 1.

    def _process_ob(self, ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ') + 2:]
        return ob   

    def render(self, mode=None):
        """渲染环境状态"""
        if not mode:
            mode = self.render_mode
        if mode == "text":
            return self.render_cache
        else:
            raise ValueError(f"Invalid render mode: {mode}")
    
    def close(self):
        self.env.close()

    
    def format_exam(self,action_info:dict):
        """        
        """
        thought, action, response = action_info['think_content'], action_info['action'], action_info['response']
        if not action :
            return {'error':True, 'format_error':True, "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: The response format should be <think>[your thought]</think>\n<answer>[your action]</answer>."}
        else:
            objects = r"([\w]+\s+\d+)"  # 匹配格式: 单词 + 空格 + 数字
            receps = r"([\w]+\s+\d+)"   # 匹配格式: 单词 + 空格 + 数字
            patterns = [
                    # 1. go to {recep}
                    rf"go\s+to\s+{receps}",
                    
                    # 2. take {obj} from {recep} 
                    rf"take\s+{objects}\s+from\s+{receps}",
                    
                    # 3. put {obj} in/on {recep}
                    rf"put\s+{objects}\s+in/on\s+{receps}",  # 修改这行，添加 in/on 选项
                    
                    # 4. open {recep}
                    rf"open\s+{receps}",
                    
                    # 6. use {obj}/{recep}
                    rf"^use \w+ \d+$", 
                    
                    # 7. clean {obj} with {recep}
                    rf"clean\s+{objects}\s+with\s+{receps}",
                    
                    # 8. heat {obj} with {recep}
                    rf"heat\s+{objects}\s+with\s+{receps}",
                    
                    # 9. cool {obj} with {recep}
                    rf"cool\s+{objects}\s+with\s+{receps}",

                    rf"inventory"
                ]
            if action :
                action_pattern = '|'.join(patterns)
                action_match = re.search(action_pattern, action.lower())
                if not action_match:
                    return {'error':True, 'format_error':True, "error_msg": "ERROR:  The model output is illegal, please check it carefully. Tips: Only one action can be performed per turn. The action should be in the following format: '1.go to recep, 2.take obj from recep, 3.put obj in/on recep, 4.open recep, 5.use obj/recep, 6.clean obj with recep, 7.heat obj with recep, 8.cool obj with recep, 9.inventory', where obj and recep are the objects and receptacles in the environment. Objects and containers must contain their designation, for example: take apple 1 from frige 1"}
            
            return {"error": False,"error_msg": ""}

    def clean_record(self):
        self.history = []
        self.data_line = None
        self.task_idx = None
        self.step_count = 0
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.num_env_steps = 0
        self.isdone = False
        self.reward = 0
        self.sub_goal = []
        self.finished_sub_goal = []
        self.cur_task_name = ""
        self.init_obs = ""
        self.goal = ""


# if __name__ == "__main__":
#     # 测试代码
#     env = AlfworldGemEnv(mode="train", max_turns=5)
#     obs, info = env.reset(seed=42)
#     print("Initial observation:", obs)
#     print("Suffix:", info["suffix"])
    
#     # 执行几个测试动作
#     test_actions = ["<answer>look</answer>", "<answer>inventory</answer>", "<answer>examine room</answer>"]
#     for action in test_actions:
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"Action: {action}")
#         print(f"Observation: {obs}")
#         print(f"Reward: {reward}, Done: {terminated}, Info: {info.get('success', False)}")
#         if terminated or truncated:
#             break
    
#     env.close()
