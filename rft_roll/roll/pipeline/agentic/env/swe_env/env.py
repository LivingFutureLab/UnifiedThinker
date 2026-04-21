import copy
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Tuple, Optional, Dict

import gymnasium as gym
from gem import Env
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.define.observation import Observation
from roll.pipeline.agentic.env.swe_env.util.repo_env import RepoClient
from roll.pipeline.agentic.env.swe_env.utils import (
    _lazy_load_jsonl_lines_spec_idx,
    write_data_json,
    pretty_print,
    Colors,
    MultiprocessSafeLogger,
)
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.random_utils import all_seed

DEBUG = True


class SWEEnv(Env, gym.Env):
    def __init__(
        self,
        render_mode: str = "text",
        max_steps: int = 50,
        max_reset_retry_times: int = 20,
        format_penalty=0.0,
        mode: str = "train",  # train, val, spec-xx
        data_path: str = "data/part_0.jsonl",
        train_idx_range: Tuple[int, int] = (0, 1e9),  # 训练集任务ID范围
        val_idx_range: Tuple[int, int] = (0, 1e9),  # 验证集任务ID范围
        tools: list[str] = [
            "swe_env/util/tools/search.py",
            "swe_env/util/tools/file_editor.py",
            "swe_env/util/tools/execute_bash.py",
            "swe_env/util/tools/finish.py",
        ],
        action_pattern="^<answer>(.*?)</answer>$",
        special_token_list=("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"),
        swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
        traj_dir: str = "./traj/trainset/",
        swe_requirment_dir: str = "/home/lixing/workspace/swe_rele/dataset/2_docker_file/250820_valset_v1_swe_bench_verified_requirment",
        base_dir: str = "./logs",
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        timeout: int = 180,
        max_env_time: float = 60 * 60,
        base_agent: str = "swe",
        sanbox_mode: str = "http",
        **kwargs,
    ):
        self.sanbox_mode = sanbox_mode
        print(f"[SWEEnv]sanbox_mode: {self.sanbox_mode}")
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.format_penalty = format_penalty
        self.base_dir = base_dir
        # print(f"*************** SWEEnv的参数 ***************")
        # print(f'swe_rex_host: {swe_rex_host}')
        # print(f'base_dir: {base_dir}')
        # print(f'max_execute_time: {max_execute_time}')
        # print(f'max_execute_retry: {max_execute_retry}')
        # print(f'timeout: {timeout}')
        # print(f"*************** SWEEnv的参数 ***************")

        # 环境信息(不变)
        self.swe_rex_host = swe_rex_host
        self.mode = mode
        self.train_idx_range = train_idx_range
        self.val_idx_range = val_idx_range
        self.max_execute_time = max_execute_time if max_execute_time else 300.0
        self.max_execute_retry = max_execute_retry if max_execute_retry else 10
        self.timeout = timeout if timeout else 180
        self.base_agent = base_agent if base_agent else "swe"
        self.data_path = data_path

        # 基本参数(不变)
        self.max_reset_retry_times = max_reset_retry_times
        self.max_steps = max_steps
        current_file_path = Path(__file__).resolve()
        self.tools = [f"{current_file_path.parent.parent}/{data_path}" for data_path in tools]
        self.traj_dir = traj_dir

        # 当前参数(会更新)
        self.retry_time = 0
        self.turn_count = 0
        self.task_idx = None
        self.history = []
        self.data_line = {}
        self.container_name = None
        self.route_key = None
        self.problem_statement = None
        self.issue = None
        self.metrics = None
        self.terminate = False
        self.truncated = False
        self.env_timeout = False
        self.env_failed = False
        self.reward = 0
        self.action_is_valid_lst = []  # 用于metric
        self.action_is_effective_lst = []  # 用于metric
        self.current_step = -1
        self.reach_max_length = False
        self.unittest_output = ""
        self.is_closed = False
        self.logger = None  # 初始化 logger 为 None

        # 时间参数
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.traj_rollout_time = 0
        self.time_start = 0
        self.max_env_time = max_env_time  # 单环境最长rollout40min, 超时则return mask。

        # 环境参数
        os.environ["SWE_REQUIRMENT_DIR"] = swe_requirment_dir

        # 数据
        print("[base_dir]", self.base_dir)
        if "part_" in self.data_path:
            # TODO:
            dataset_name = self.data_path.replace("/part_0.jsonl", f"")
        else:
            dataset_name = self.data_path
        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"

        self.dataset = GlobalDataset.options(
            name=f"{self.mode}_{dataset_name}", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote(dataset_name=dataset_name, mode=global_dataset_mode)

        # 这里开启做数据过滤，待进一步补充更多
        # TODO 不同环境的基本字段对齐
        # 通过index保留特定条数
        data_ranges = self.val_idx_range if self.mode == "val" else self.train_idx_range
        ray.get(
            self.dataset.filter.remote(
                filter_name="filter_idx_range", function=lambda x: data_ranges[0] <= int(x["idx"]) <= data_ranges[1]
            )
        ) 
        # TODO swe带上pass
        # ray.get(
        #     self.dataset.filter.remote(
        #         filter_name="filter_pass_range", function=lambda x: pass_low_threshold <= int(x["pass_ratio"]) <= pass_high_threshold
        #     )
        # )


        self.dataset_manager = GlobalDatasetManager.options(
            name=f"{self.mode}_dataset_manager", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        

    def get_task_suffix(self) -> Any:
        problem_statement, issue = self.get_instruction()
        return problem_statement

    def get_task_idx_and_data(self, seed):
        data_item: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data_item is None:
            return None, None
        idx = data_item["idx"]
        if self.mode == "val":
            print(
                f"[GEN_IDX]mode: {self.mode}, self.val_idx_range: {self.val_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        elif self.mode == "train":
            print(
                f"[GEN_IDX]mode: {self.mode}, self.train_idx_range: {self.train_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        else:
            print(
                f"[GEN_IDX][Attention] mode: {self.mode}, val_idx_range: {self.val_idx_range}, train_idx_range: {self.train_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        return idx, data_item

    def gen_route_key(self, data_line):
        if "docker_image" in data_line:
            route_key = f"{self.task_idx}_{data_line['docker_image']}"
        elif "docker_image" not in data_line:
            route_key = f"{self.task_idx}"
        return route_key

    def get_instruction(self):
        problem_statement = self.data_line["problem_statement"]
        try:
            issue = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", problem_statement, re.DOTALL).group(1)  # r2e-gym trainset
        except:
            issue = problem_statement  # swe-bench-verified
        return problem_statement, issue

    def reset(self, seed=None, step=None):
        st = time.time()
        self.time_start = st
        self.global_step = step
        self.clean_record()

        # gen task_idx and load data_line
        task_idx, data_line = self.get_task_idx_and_data(seed)
        if data_line is None:
            return None, {}

        self.task_idx = task_idx
        self.data_line = data_line

        # init logger - 只在第一次创建，后续 reset 只更新文件路径
        time_str = time.strftime("%m%d%H%M%S%f", time.localtime())
        log_path = os.path.join(
            self.base_dir,
            f"log/step{step}/env",
            f"{self.mode}_seed{seed}_idx{self.task_idx}-{time_str}-{time.time_ns()}.log",
        )
        
        if self.logger is None:
            # 第一次创建 logger
            self.logger = MultiprocessSafeLogger(path=log_path)
        else:
            # 后续 reset 只更新文件路径
            self.logger.update_log_path(path=log_path)
        
        self.logger.info(f"start reset, task_idx: {self.task_idx}")

        self.repo_env = RepoClient(
            logger=self.logger,
            max_execute_time=self.max_execute_time,
            max_execute_retry=self.max_execute_retry,
            timeout=self.timeout,
            max_env_time=self.max_env_time,
            sanbox_mode=self.sanbox_mode,
            swe_rex_host=self.swe_rex_host,
        )
        data_line["swe_rex_host"] = self.swe_rex_host

        self.logger.info(
            f"\n\n*************** 环境初始化 start reset, task_idx: {self.task_idx}, seed: {seed}, global_step: {self.global_step}***************"
        )
        while (
            self.retry_times < self.max_reset_retry_times and (time.time() - st) < self.max_env_time
        ):  # Attention: 这里必须成功，否则会failed
            if self.retry_times == 0:
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} [ENV][RESET]start reset, task_idx: {self.task_idx}, global_step: {self.global_step}'
                )
                self.logger.info(
                    f"[ENV][RESET]start reset, task_idx: {self.task_idx}, global_step: {self.global_step}"
                )
            # init docker_runtime
            # TODO：这里后续可以在环境reset里写多进程reset，哪个先成功就用哪个
            reset_info = self.repo_env.reset(
                data_line,
                max_execute_time=self.max_execute_time,
                max_execute_retry=self.max_execute_retry,
                timeout=self.timeout,
            )  # {'container_name':ERROR/SUCCESS,'setup_env_result':ERROR/SUCCESS,"reset_retry_times": int, "route_key": self.route_key}
            self.container_name = reset_info.get("container_name", None)
            self.route_key = reset_info.get("route_key", None)
            self.retry_times += reset_info.get("retry_times", 1)
            self.setup_env_result = reset_info.get("setup_env_result", None)

            self.logger.info(
                f"[ENV][RESET](retry_times: {self.retry_times})reset_info: {reset_info}, global_step: {self.global_step}"
            )
            print(
                f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][ENV][RESET](retry_times: {self.retry_times})reset_info: {reset_info}'
            )
            if reset_info["state"] == "success":
                break
            else:
                print(
                    f"[ENV][RESET](retry_times: {self.retry_times})reset_info: {reset_info}, global_step: {self.global_step}"
                )

        if not self.container_name or "ERROR" in self.container_name:
            print(
                f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][ENV][RESET][ERROR]reset failed (retry_times: {self.retry_times})'
            )
            self.logger.info(
                f"[ENV][RESET][ERROR]reset {self.task_idx} failed (retry_times: {self.retry_times}, used {(round(time.time() - st, 4))/60} mins)"
            )
            self.env_failed = True
            return "error", {"suffix": "error"}

        self.data_line["container_name"] = self.container_name

        # add tools to repo_env
        self.repo_env.add_commands(self.tools)

        # get_instruction
        self.problem_statement, self.issue = self.get_instruction()

        # record time
        self.traj_reset_time = round(time.time() - st, 4)

        self.history.append({"role": "user", "content": self.problem_statement})

        self.allowed_cmds = self.repo_env.get_available_cmds()

        if self.base_agent == "iflow":
            self.system_prompt = self.repo_env.get_system_prompt_in_iflow()
        else:
            self.system_prompt = ""

        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]finish reset, task_idx: {self.task_idx}, retry_times: {self.retry_times}, retry_time: {self.traj_reset_time}s'
        )
        self.logger.info(
            f"[ENV][RESET]finish reset, task_idx: {self.task_idx}, retry_times: {self.retry_times}, time: {self.traj_reset_time}s, route_key: {self.route_key}, container_name: {self.container_name}"
        )
        return self.problem_statement, {
            "suffix": self.issue,
            "system_prompt": self.system_prompt,
        }  # TODO suffix和obs的区别

    def step(self, action: str):
        """
        @input:
            action: <answer>Right</answer>
        @output:
            [obs] At turn 1, you moved Down, which is effective.
            [reward] 0.0
            [terminate] False
            [truncated] False
            [info] {'suffix': 'Here is the current state of the FrozenLake:\n____\n_OP_\n___O\nGO__\n', 'metrics': {'action_is_effective': True, 'action_is_valid': True, 'success': False, 'format_penalty': 0.0}, 'action': 1, 'action_content': 'Down', 'think_content': ''}
        """
        # print(f'\n\n-------------- SWEEnv.step(start) --------------')
        # print(f'\n\n[ENV][STEP][action]{action}')
        if action == "max_length":
            self.reward, self.unittest_output = self.calculate_reward()
            print(
                f"[接近max_length]task_idx: {self.task_idx}, 当前reward: {self.reward}, unittest_output: \n{self.unittest_output}"
            )
            return self.unittest_output, self.reward, False, False, {}

        st = time.time()
        self.turn_count += 1

        self.logger.info(
            f"*************** 环境交互输入 (task_idx: {self.task_idx})(turn: {self.turn_count})(global_step: {self.global_step})] *******************\n\n"
            f"********* model_input ******\n{[action]}\n"
        )
        obs = ""
        info = {"suffix": "", "metrics": ""}
        bash_output, exit_code, execute_time = "", "", 0
        action_is_valid, action_is_effective = False, False

        # update history with response
        self.history.append({"role": "assistant_original", "content": f"{action}"})

        # 因交互时间过长 或者 环境初始化失败t
        self.traj_total_time = round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4)
        self.traj_rollout_time = round(time.time() - self.time_start, 4)

        if self.env_failed or not self.container_name:
            self.truncated = True
            self.metrics = {
                "env_timeout": False,
                "env_failed": True,
                "reach_max_length": False,
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": (
                    round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                    if self.action_is_valid_lst
                    else 0.0
                ),
                "action_is_effective": (
                    round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                    if self.action_is_effective_lst
                    else 0.0
                ),
                "turn_count": self.turn_count,
                "retry_times": self.retry_times,
                "traj_reset_time": self.traj_reset_time,
                "traj_step_time": round(self.traj_step_time, 4),
                "traj_reward_time": self.traj_reward_time,
                "traj_total_time": self.traj_total_time,
                "traj_rollout_time": self.traj_rollout_time,
                "task_idx": self.task_idx,  # for rollout_log
            }
            info["metrics"] = self.metrics
            obs = f"ERROR: The container name is {self.container_name}"
            print(
                f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][ENV][STEP][容器初始化失败] task_idx: {self.task_idx}, route_key: {self.route_key}, docker_image: {self.data_line.get("docker_image", "")}'
            )
            # self.logger.info(f'[ENV][STEP][ERROR IN ENV] 容器初始化失败 (task_idx: {self.task_idx}, route_key: {self.route_key}, docker_image: {self.data_line.get("docker_image", "")})')
            self.logger.info(
                f"[ENV][STEP]******************* 环境交互输出【容器初始化失败】 *******************\nobs: {obs}\nreward: {self.reward}\nterminate: {self.terminate}\ntruncated: {self.truncated}\ninfo: {info}"
            )
            self.close()
            return obs, self.reward, self.terminate, self.truncated, info
        elif time.time() - self.time_start > self.max_env_time:
            self.logger.info(
                f'[ENV][STEP][ERROR IN ENV] 交互时间超过{(self.max_env_time)/60}min (task_idx: {self.task_idx}, route_key: {self.route_key}, docker_image: {self.data_line.get("docker_image", "")}, current_step: {self.turn_count})'
            )
            self.truncated = True
            self.env_timeout = True
            self.metrics = {
                "env_timeout": True,
                "env_failed": False,
                "reach_max_length": False,
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": (
                    round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                    if self.action_is_valid_lst
                    else 0.0
                ),
                "action_is_effective": (
                    round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                    if self.action_is_effective_lst
                    else 0.0
                ),
                "turn_count": self.turn_count,
                "retry_times": self.retry_times,
                "traj_reset_time": self.traj_reset_time,
                "traj_step_time": round(self.traj_step_time, 4),
                "traj_reward_time": self.traj_reward_time,
                "traj_total_time": self.traj_total_time,
                "traj_rollout_time": self.traj_rollout_time,
                "task_idx": self.task_idx,  # for rollout_log
            }  # TODO：
            info = {
                "suffix": obs,
                "metrics": self.metrics,
            }
            print(
                f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][ENV][STEP][ENV_TIMEOUT]交互时间超过{(self.max_env_time)/60}min (task_idx: {self.task_idx}, route_key: {self.route_key}, docker_image: {self.data_line.get("docker_image", "")}, current_step: {self.turn_count})'
            )
            info["metrics"]["env_timeout"] = True  # 增加了一个这个
            obs = f"ERROR: The command took too long to execute (>{self.max_env_time}s)"
            self.logger.info(
                f"[ENV][STEP]******************* 环境交互输出【超时】 *******************\nobs: {obs}\nreward: {self.reward}\nterminate: {self.terminate}\ntruncated: {self.truncated}\ninfo: {info}"
            )
            self.close()
            return obs, self.reward, self.terminate, self.truncated, info

        # split think content
        if "</think>" in action:
            action = action.split("</think>")[-1].strip()

        # parse action from response & format exam
        action_info = self.parse_action(action)
        format_info = self.format_exam(action_info)  # error, error_msg
        self.logger.info(
            f"[ENV][STEP][解析Action]\n"
            f'****** think_content ******\n{action_info["think_content"]}\n\n'
            f'****** action_content ******\n{action_info["action_content"]}\n\n'
        )
        self.logger.info(f"[ENV][STEP][检查格式]{format_info}")
        if DEBUG:
            print(f"[ENV][STEP][解析action]{action_info}\n[ENV][STEP][检查格式]{format_info}")

        # run action and get obs
        if not format_info["error"]:
            bash_output, exit_code, execute_time = self.repo_env.run_action(
                action_info["action"], timeout=180, base_agent=self.base_agent
            )  # Action object
            obs = str(Observation(bash_output, exit_code, action_info["action"]))
        else:
            obs = format_info.get("error_msg", "")
        # self.logger.info(f'[ENV][STEP][obs]run action and get obs: \n{obs}')

        # invalid action
        try:
            if not ("Invalid Action" in str(exit_code)) and not format_info["error"]:  # 无效动作
                action_is_valid = True
            if exit_code == "0":  # 动作执行成功：
                action_is_effective = True
        except Exception as e:
            print(f"[DEBUG][error in action valid]{e}")
            action_is_valid = False
            action_is_effective = False

        # finish, submit
        if (
            "finish" in action_info["action"].function_name.lower()
            or "submit" in action_info["action"].function_name.lower()
            or "<function=submit />" in action
        ):  # 任务完成
            self.terminate = True
            action_is_valid, action_is_effective = True, True
            self.logger.info(f"[ENV][STEP][terminate]主动结束")
        elif action == "max_length":
            self.logger.info(f"[ENV][STEP][truncated]超过最大Token数强制结束")
            self.reach_max_length = True
            action_is_valid, action_is_effective = True, True
        # 超过最大轮数强制结束
        if not self.terminate and self.turn_count >= self.max_steps:
            self.truncated = True
            self.logger.info(f"[ENV][STEP][truncated]超过最大轮数强制结束")
        # 计算reward
        if self.terminate or self.truncated or self.reach_max_length:
            st_reward = time.time()
            print(
                f"task {self.task_idx} start calculate reward ... (terminate: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length})"
            )
            self.reward, self.unittest_output = self.calculate_reward()
            self.traj_reward_time = round(time.time() - st_reward, 4)
            self.traj_total_time = round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4)
            self.traj_rollout_time = round(time.time() - self.time_start, 4)
            self.logger.info(f"[ENV][STEP][reward]计算reward: {self.reward}")
            print(f"end calculate reward ... {self.reward}, unittest_output: \n{self.unittest_output}")
            obs = self.unittest_output

        self.history.append(
            {
                "role": "assistant",
                "content": f'{action_info["action_content"]}',
                "action_is_valid": action_is_valid,
                "action_is_effective": action_is_effective,
            }
        )
        self.history.append({"role": "user", "content": f"{obs}"})

        self.action_is_valid_lst.append(1 if action_is_valid else 0)
        self.action_is_effective_lst.append(1 if action_is_effective else 0)

        self.traj_step_time += round(time.time() - st, 4) - self.traj_reward_time

        self.metrics = {
            "env_timeout": self.env_timeout,
            "env_failed": self.env_failed,
            "reach_max_length": self.reach_max_length,
            "reward": self.reward,
            "success": self.terminate,
            "truncated": self.truncated,
            "format_penalty": self.format_penalty,
            "action_is_valid": (
                round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                if self.action_is_valid_lst
                else 0.0
            ),
            "action_is_effective": (
                round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                if self.action_is_effective_lst
                else 0.0
            ),
            "turn_count": self.turn_count,
            "retry_times": self.retry_times,
            "traj_reset_time": self.traj_reset_time,
            "traj_step_time": round(self.traj_step_time, 4),
            "traj_reward_time": self.traj_reward_time,
            "traj_total_time": self.traj_total_time,
            "traj_rollout_time": self.traj_rollout_time,
            "task_idx": self.task_idx,  # for rollout_log
        }  # TODO：
        info = {
            "suffix": obs,
            "metrics": self.metrics,
        }
        # print('-------------------------------------------------')
        # print(f'[DEBUG][obs]{obs}')
        # print(f'[DEBUG][reward]{self.reward}')
        # print(f'[DEBUG][terminate]{self.terminate}')
        # print(f'[DEBUG][truncated]{self.truncated}')
        # print(f'[DEBUG][metrics]{self.metrics}')
        # print(f'[DEBUG][action_is_valid]{action_is_valid}')
        # print(f'[DEBUG][action_is_effective]{action_is_effective}')
        # print(f'-------------- SWEEnv.step(end) -----------------')
        self.logger.info(
            f"\n*************** 环境交互输出【正常】***************\nstep: {self.turn_count}\nobs: {obs}\nreward: {self.reward}\nterminate: {self.terminate}\ntruncated: {self.truncated}\ninfo: {info}"
        )
        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}][ENV STEP][task_idx:{self.task_idx}][Step:{self.turn_count}][Rollout时间消耗]{round((time.time() - self.time_start)/60,2)}min. [服务交互消耗]{round((self.traj_total_time)/60,2)}min. [Metrics]{self.metrics} (route_key: {self.route_key})'
        )
        if self.terminate or self.truncated or self.reach_max_length:
            self.close()
        return obs, self.reward, self.terminate, self.truncated, info

    def parse_action(self, response_text):
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action_obj = Action.from_string(action)
        # print(f'[ATTENTION][response_text]{response_text}\n[before] action: {action}\n[after]action_obj: {action_obj}')

        action_info = {
            "response": response_text,
            "action": action_obj,  # Action object
            "action_content": action,  # string
            "think_content": thought,  # string
        }
        return action_info

    def format_exam(self, action_info: dict):
        """ """
        thought, action, response = action_info["think_content"], action_info["action"], action_info["action_content"]

        # format error type1
        action_dict = action.to_dict()
        if action_dict["function"] == "":
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1><parameter=param1>xxx</parameter><parameter=param2>xxx</parameter></function>'.",
            }
        # format error type2
        elif "<parameter>" in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=xxxx>', not '<parameter>xxxx>.'",
            }
        elif "<function>" in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>', not '<function>func1>.'",
            }
        elif "<parameter" in response and "</parameter>" not in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=param1>xxx</parameter>'. Do not forget to close the parameter tag.",
            }
        elif "<function" in response and "</function>" not in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>xxxx</function>'. Do not forget to close the function tag.",
            }
        elif not action:
            return {"error": True, "error_msg": "ERROR: The model output is illegal, please check it carefully."}
        # format error type3: function name
        function_name = action_dict["function"]
        if function_name not in self.allowed_cmds:
            return {
                "error": True,
                "error_msg": f"Invalid Action: input action must be one of allowed actions. Allowed actions: {self.allowed_cmds}. Current input action: {function_name}. ",
            }

        return {"error": False, "error_msg": ""}

    def render(self, mode=None):
        # 源frozen_lake的render函数，将环境的内部数值状态转换为直观的文本或图像表示，用在get_instruction中
        # 这里应该不需要写
        pass

    def calculate_reward(self):
        self.reward, self.unittest_output = self.repo_env.calculate_reward()
        self.unittest_output = "UNITTEST OUTPUT: \n" + self.unittest_output
        return self.reward, self.unittest_output

    def sample_random_action(self):
        actions = []
        # 执行正确动作
        actions.append(
            "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        )
        actions.append(
            "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        )
        actions.append(
            "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        )
        # 执行不允许的动作（exit code: 1）
        actions.append(
            "the issue:\n\n<function=execute_bash>\n  <parameter=cmd>cd /testbed && python reproduce_issue.py</parameter>\n</function>"
        )
        # 执行错误动作（exit code: 2）
        actions.append("<function=execute_bash>\n  <parameter=cmd>python reproduce_issue.py</parameter>\n</function>")
        # 提交
        actions.append(
            'tness.\n\n<function=finish>\n  <parameter=command>submit</parameter>\n  <parameter=result>I have successfully implemented a fix for the GitHub issue "Callback Receives Unexpected Argument When No Result is Provided" in the `@return_future` decorator.\n\n**Changes Made:**\n1. Added `import inspect` to `/testbed/tornado/concurrent.py`\n2. Modified the `run_callback` function to intelligently handle callbacks when no result is provided:\n   - When `result is None`, inspect the callback signature\n   - If callback accepts no parameters, call it without arguments\n   - Otherwise, call it with the `None` result as before\n   - Includes robust fallback mechanism for cases where signature inspection fails\n\n**Key Benefits:**\n- \u2705 Fixes the original issue: callbacks expecting no arguments work correctly\n- \u2705 Maintains backward compatibility: all existing tests pass\n- \u2705 Handles edge cases: works with lambdas, default parameters, *args, **kwargs\n- \u2705 Robust: graceful fallback when signature inspection fails\n- \u2705 Minimal impact: only changes behavior when result is `None`\n\nThe fix ensures that when a `@return_future` decorated function calls its callback without arguments, the client callback is invoked appropriately based on its signature, resolving the TypeError while preserving all existing functionality.</parameter>\n</function>'
        )
        return random.choice(actions)

    def close(self, stop_reason=None):
        print(f"*************** task_idx:{self.task_idx} SWEEnv.close(start) ***************")
        if self.is_closed:
            print(f"[重复close]task_idx: {self.task_idx}")
            return

        print(f"*************** task_idx:{self.task_idx} SWEEnv.close(write_data_json) ***************")
        if self.env_timeout:
            stop_reason = "env_timeout"
        elif self.env_failed:
            stop_reason = "env_failed"
        elif self.truncated:
            stop_reason = "truncated"
        elif self.terminate:
            stop_reason = "terminate"
        elif stop_reason:
            stop_reason = stop_reason
        else:
            stop_reason = "unknown"
        self.history.append({"role": "reward", "content": f"{self.unittest_output}", "reward": self.reward})
        save = {
            "env_timeout": self.env_timeout,
            "env_failed": self.env_failed,
            "reach_max_length": self.reach_max_length,
            "global_step": self.global_step,
            "task_idx": self.task_idx,
            "reward": self.reward,
            "unittest_output": self.unittest_output,
            "terminate": self.terminate,
            "truncated": self.truncated,
            "stop_reason": stop_reason,
            "container_name": self.container_name,
            "route_key": self.route_key,
            # "problem_statement": self.problem_statement,
            "retry_times": self.retry_times,
            "docker_image": self.data_line.get("docker_image", ""),
            "history": self.history,
            "metrics": self.metrics,
            "retry_times": self.retry_times,
        }
        os.makedirs(self.traj_dir, exist_ok=True)
        valid_score = (
            round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 2)
            if self.action_is_valid_lst
            else 0.0
        )
        effective_score = (
            round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 2)
            if self.action_is_effective_lst
            else 0.0
        )
        time_str = time.strftime("%m%d%H%M%S", time.localtime())
        log_path = os.path.join(
            self.traj_dir,
            f"step{self.global_step}/env_traj",
            f"re{self.reward}-{stop_reason}-{self.task_idx}-{time_str}-v{valid_score}_e{effective_score}_tc{self.turn_count}_time{self.traj_total_time}.json",
        )
        # write_data_json(save,log_path)
        # close
        self.repo_env.close()
        # 这里不能clean_record，否则返回就不对了。
        print(
            f"[ENV CLOSE][global_step:{self.global_step}][task_idx:{self.task_idx}][Step:{self.turn_count}][Rollout时间消耗]{round((time.time() - self.time_start)/60,2)}min. [服务交互消耗]{round((self.traj_total_time)/60,2)}min. [Metrics]{self.metrics} (route_key: {self.route_key})"
        )
        self.logger.info(
            f"\n*************** 【释放环境({stop_reason})】 ***************\n[global_step:{self.global_step}][task_idx:{self.task_idx}][Step:{self.turn_count}][Rollout时间消耗]{round((time.time() - self.time_start)/60,2)}min. [服务交互消耗]{round((self.traj_total_time)/60,2)}min. [Metrics]{self.metrics} (route_key: {self.route_key}), log: {log_path}"
        )
        self.logger.save()
        if self.max_env_time == 0:
            self.max_env_time = 60 * 40  # 单环境最长rollout40min, 超时则return mask。
        self.history = []
        self.data_line = {}
        self.is_closed = True

    def clean_record(self):
        print(f"[DEBUG][参数重置 ...]")
        self.history = []
        self.data_line = {}
        self.container_name = None
        self.route_key = None
        self.task_idx = None
        self.turn_count = 0
        self.retry_times = 0
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_total_time = 0
        self.problem_statement, self.issue = "", ""
        self.reward, self.terminate, self.truncated = 0, False, False
        self.env_timeout = False
        self.env_failed = False
        if self.max_env_time == 0:
            self.max_env_time = 60 * 40  # 单环境最长rollout40min, 超时则return mask
        self.env_timeout = False
        self.env_failed = False
        self.reach_max_length = False
        self.unittest_output = ""
        self.is_closed = False

    def get_history(self):
        return self.history

    def get_key_params(self):
        return {
            "task_idx": self.task_idx,
            "container_name": self.container_name,
            "route_key": self.route_key,
            "problem_statement": self.problem_statement,
            "retry_times": self.retry_times,
            "docker_image": self.data_line.get("docker_image", ""),
        }


if __name__ == "__main__":

    """
    env.step 返回的info中必须有suffix字段
    """
    # data = load_data_json('/home/lixing/workspace/future_update/ROLL_version/ScaleAligner/logs/dataset/98_traj.json')

    # try:
    env = SWEEnv(
        mode="spec-0",
        # data_path="/home/lixing/workspace/swe_rele/dataset/2_docker_file/250814_trainset_v1_r2e_lite_vpc_4578_split100/part_0.jsonl",
        data_path="/home/lixing/workspace/swe_rele/dataset/2_docker_file/250814_valset_v1_swe_bench_verified_vpc_500_split100/part_0.jsonl",
        # data_path='/home/lixing/workspace/process/dataset/swe/docker_image/swe-rebench-repo-v1-sample10-r2e.jsonl',
        # data_path = '/home/lixing/workspace/process/dataset/swe/250922_valset_v2_swe_bench_verified_vpc_500_split100-iflow/part_0.jsonl',
        train_idx_range=(0, 4577),
        val_idx_range=(0, 128),
        max_steps=50,
        max_reset_retry_times=20,
        format_penalty=0.0,
        tools=[
            "swe_env/util/tools/file_editor.py",
            "swe_env/util/tools/execute_bash.py",
            "swe_env/util/tools/search.py",
            "swe_env/util/tools/finish.py",
        ],
        # tools = [
        #     "list_directory","read_file","write_file","replace","multi_edit","search_file_content","glob","web_search","web_fetch","todo_write","todo_read","run_shell_command"
        # ],
        action_pattern="^<answer>(.*?)</answer>$",
        special_token_list=("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"),
        swe_rex_host="https://xrl-aliyun.alibaba-inc.com/swe-rex/docker",
        max_env_time=60 * 40,
        # base_agent='iflow' # choice from swe or iflow
        base_agent="swe",  # choice from swe or iflow
    )
    obs, info = env.reset(seed=42, step=0)
    pretty_print(Colors.PINK, f"\n[observation]", f"{obs}")
    pretty_print(Colors.PINK, f"\n[info]", f"{info}")
    keyboard_lst = [
        """The error has been resolved, and the separability matrix now reflects the correct behavior for nested compound models. The output is as expected, showing the correct separability for each input and output.

I'll now finalize this task.

<function=write_file>
<parameter=file_path>/test.txt</parameter>
<parameter=content>Hello, world!</parameter>
</function>""",
        """I'll now finalize this task.xxx
<function=read_file>
<parameter=absolute_path>/test.txt</parameter>
</function>
""",
    ]

    keyboard_lst = [
        """ Let me simplify and run the script directly:\n<function=file_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/reproduce_issue.py</parameter>\n<parameter=old_str>def main():\n    print(\"=\" * 60)\n    print(\"Reproducing NPY_BLAS_ORDER empty string issue\")\n    print(\"=\" * 60)\n    \n    # Create clean environment for testing\n    with tempfile.TemporaryDirectory() as temp_dir:\n        original_cwd = os.getcwd()\n        os.chdir(temp_dir)\n        \n        try:\n            success = test_empty_string_issue()\n            print(f\"\\nTest result: {'PASSED' if success else 'FAILED'}\")\n            \n            if not success:\n                sys.exit(1)\n                \n        finally:\n            os.chdir(original_cwd)\n\nif __name__ == \"__main__\":\n    main()</parameter>\n<parameter=new_str>def main():\n    print(\"=\" * 60)\n    print(\"Reproducing NPY_BLAS_ORDER empty string issue\")\n    print(\"=\" * 60)\n    \n    try:\n        success = test_empty_string_issue()\n        print(f\"\\nTest result: {'PASSED' if success else 'FAILED'}\")\n        \n        if not success:\n            sys.exit(1)\n                \n    except Exception as e:\n        print(f\"Error occurred: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()</parameter>\n</function\n\n<function=finish>\n<parameter=command>submit</parameter>\n<parameter=result>Test: Reproducing NPY_BLAS_ORDER empty string issue\n\nI need to understand the current behavior. Let me examine the issue more closely by looking at what happens when the environment variable is empty:</function>"""
    ]
    # keyboard_lst = ["""<function=finish>
    # <parameter=command>submit</parameter>
    # <parameter=result>I have successfully implemented the necessary changes to fix the issue with qop options not being quoted in Digest Auth. Here's a summary of what was done:

    # ## Issue Analysis
    # The GitHub issue reported that the `qop-options` directive in Digest Authentication should be quoted with double quotes according to RFC2617, but the requests library was outputting `qop=auth` instead of `qop="auth"`.

    # ## Changes Made
    # I made a minimal change to the `/testbed/requests/auth.py` file in the `build_digest_header` method of the `HTTPDigestAuth` class:

    # **File:** `/testbed/requests/auth.py`
    # **Line 147:** Changed from:
    # ```python
    # base += ', qop=auth, nc=%s, cnonce="%s"' % (ncvalue, cnonce)
    # ```
    # to:
    # ```python
    # base += ', qop="auth", nc=%s, cnonce="%s"' % (ncvalue, cnonce)
    # ```

    # ## Validation
    # 1. **Created reproduction script** (`reproduce_issue.py`) that confirmed the issue existed
    # 2. **Applied the fix** by adding quotes around the "auth" value in the qop parameter
    # 3. **Verified the fix** with multiple test scenarios:
    #    - Single qop value 'auth'
    #    - Multiple qop values 'auth,auth-int'
    #    - Multiple qop values 'auth-int,auth'
    #    - No qop value (should not include qop in header)
    # 4. **Ran existing tests** to ensure no regressions - all digest-related tests pass

    # ## Result
    # The change ensures that the qop value in Digest Authentication headers is properly quoted as `qop="auth"` instead of `qop=auth`, making it compliant with RFC2617 and compatible with server-side implementations that are sensitive to this difference (as mentioned in the issue).

    # The fix is minimal, targeted, and maintains backward compatibility while improving RFC compliance.</parameter>
    # </function>
    # """]

    # for i in range(len(data)):
    for i in range(len(keyboard_lst)):
        print(f"\n[i]{i}")
        # keyboard = input("Enter action: ")
        # if keyboard == "q":
        # break
        # keyboard = """ Let me simplify and run the script directly:\n<function=file_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/reproduce_issue.py</parameter>\n<parameter=old_str>def main():\n    print(\"=\" * 60)\n    print(\"Reproducing NPY_BLAS_ORDER empty string issue\")\n    print(\"=\" * 60)\n    \n    # Create clean environment for testing\n    with tempfile.TemporaryDirectory() as temp_dir:\n        original_cwd = os.getcwd()\n        os.chdir(temp_dir)\n        \n        try:\n            success = test_empty_string_issue()\n            print(f\"\\nTest result: {'PASSED' if success else 'FAILED'}\")\n            \n            if not success:\n                sys.exit(1)\n                \n        finally:\n            os.chdir(original_cwd)\n\nif __name__ == \"__main__\":\n    main()</parameter>\n<parameter=new_str>def main():\n    print(\"=\" * 60)\n    print(\"Reproducing NPY_BLAS_ORDER empty string issue\")\n    print(\"=\" * 60)\n    \n    try:\n        success = test_empty_string_issue()\n        print(f\"\\nTest result: {'PASSED' if success else 'FAILED'}\")\n        \n        if not success:\n            sys.exit(1)\n                \n    except Exception as e:\n        print(f\"Error occurred: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()</parameter>\n</function\n\n<function=finish>\n<parameter=command>submit</parameter>\n<parameter=result>Test: Reproducing NPY_BLAS_ORDER empty string issue\n\nI need to understand the current behavior. Let me examine the issue more closely by looking at what happens when the environment variable is empty:</function>"""

        # if data['history'][i]['role'] == 'user': continue
        # elif data['history'][i]['role'] == 'assistant':continue
        # keyboard = data['history'][i]['content']
        # print(f'[DEBUG][keyboard]{keyboard}')
        obs, reward, terminate, truncated, info = env.step(keyboard_lst[i])
        pretty_print(Colors.PINK, f"\n[observation]", f"{obs}")
        pretty_print(Colors.PINK, f"\n[reward]", f"{reward}")
        pretty_print(Colors.PINK, f"\n[terminate]", f"{terminate}")
        pretty_print(Colors.PINK, f"\n[truncated]", f"{truncated}")
        pretty_print(Colors.PINK, f"\n[info]", f"{info}")
        history = env.get_history()
        # pretty_print(Colors.PINK, f'\n[history]',f'{history}')
        if terminate or truncated:
            break
    # finally:
    env.close()
