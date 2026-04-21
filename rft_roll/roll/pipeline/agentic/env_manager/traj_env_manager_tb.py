import copy
import json
import os
import random
import traceback
import time
from contextlib import nullcontext
from datetime import datetime
from threading import Lock
from typing import Optional, List

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env.terminal_env.terminal_env import TerminalBenchEnv
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.utils.constants import GenerateStopReason, EpisodeStopReason
from roll.utils.functionals import aggregate_metrics, pad_to_length
from roll.utils.logging import get_logger


class TrajEnvManagerTB(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler
        self.save_dir = self.pipeline_config.rollout_dump_dir if self.pipeline_config.rollout_dump_dir else "trajectory_logs"
        self.save_dir = self.save_dir + "_TEMP"
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.reset_status = False
        self.failure_mode = None
        self.stop_reason = None
        
        self.use_thread_lock = self.env_config.get("use_thread_lock", False)
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        group_id = self.env_config['group_id']
        num_env_groups = self.worker_config.num_env_groups

        if "seed" in self.env_config['config']:
            self.env_config['config']["seed"] = self.env_config['group_seed']
        try:
            self.env = TerminalBenchEnv(group_id = group_id, num_env_groups = num_env_groups, **self.env_config['config'])
            self.env_initialization_failed = False
        except Exception as e:
            self.logger.error(f"[ENV_INIT] Failed! - Environment initialization failed for group {group_id}: {str(e)}")
            self.logger.warning(f"[ENV_INIT] Skipping this environment - Environment will be marked as unavailable")
            self.env = None
            self.env_initialization_failed = True
        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        # self.agent_system_template = self.cfg_template["agent_system_template"]
        # self.agent_template = self.cfg_template["agent_template"]

        # if self.env_config["env_id"] == 0:
        #     self.logger.info(f"agent_system_template: {self.agent_system_template}")
        #     self.logger.info(f"agent_template: {self.agent_template}")

        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )
        self.env_instance_start_time = None
        self.env_instance_end_time = None

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        
        rd = random.randint(1, 30)
        time.sleep(rd)
        rollout_cache = self.reset()
        start_step = self.current_step
        
        if self.reset_status:
            self.logger.error(f"[ROLLOUT_LOOP] Failed! - due to sandbox initialization failure...")
            self.stop_reason = EpisodeStopReason.SANDBOX_INIT_FAILED
            rollout = self._create_placeholder_rollout(self.episode_id, "sandbox_initialization_failed")
            rollout.meta_info["drop_flag"] = True
            ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
            self.env.close()
            time.sleep(60)
            rollout_cache = self.reset()
            start_step = self.current_step
        
        self.logger.info(f"[ROLLOUT_LOOP] START - Group:{self.env_config['group_id']}, Env:{self.env_config['env_id']}, Episode:{self.episode_id},  Steps:{start_step}, Task_name: {self.env.task_name}, Sandbox_ip: {getattr(self.env.sandbox_util, 'sandbox_ip', 'N/A') if hasattr(self.env, 'sandbox_util') and self.env.sandbox_util else 'N/A'},sandbox_id: {getattr(self.env.sandbox_util, 'sandbox_id', 'N/A') if hasattr(self.env, 'sandbox_util') and self.env.sandbox_util else 'N/A'}, Seed:{self.group_seed}")
        

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}
        max_reset_retries = 0
        
        while self.running and rollout_cache is not None:
            
            if self.reset_status:
                max_reset_retries += 1
                self.logger.error(f"[ROLLOUT_LOOP] Failed! - due to sandbox initialization failure...")
                self.stop_reason = EpisodeStopReason.SANDBOX_INIT_FAILED
                rollout = self._create_placeholder_rollout(self.episode_id, "sandbox_initialization_failed")
                rollout.meta_info["drop_flag"] = True
                time.sleep(60)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
                self.env.close()
                if max_reset_retries > 3:
                    backoff_time = min(3600, 600 * max_reset_retries)
                    self.logger.warning(f"[ROLLOUT_LOOP] Avoidance mode - Backing off for {backoff_time}s (retry #{max_reset_retries})")
                    time.sleep(backoff_time)
                else:
                    time.sleep(60)
                rollout_cache = self.reset()
                start_step = self.current_step
                continue
            
            max_reset_retries = 0
            
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            if stop_reason == GenerateStopReason.MAX_LENGTH:
                self.stop_reason = EpisodeStopReason.MAX_LENGTH
            elif stop_reason == GenerateStopReason.ABORT:
                self.stop_reason = EpisodeStopReason.ABORT
            elif stop_reason == GenerateStopReason.NO_SYSTEM_PROMPT:
                self.stop_reason = EpisodeStopReason.NO_SYSTEM_PROMPT
           

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

    
            if self.stop_reason is None and rollout_cache.terminated:
                self.stop_reason = EpisodeStopReason.FINISH
            
                
            if self.running and (rollout_cache.terminated or stop_reason != GenerateStopReason.FINISH or self.reset_status):
                if 'observation' in rollout_cache.history[-1] and len(rollout_cache.history) > 1:
                    episode_metrics = rollout_cache.history[-2].get('metrics', {}) if rollout_cache.history else {}
                    success = episode_metrics.get('success', False)
                    reward = episode_metrics.get('raw_reward', 0)
                else:
                    episode_metrics = rollout_cache.history[-1].get('metrics', {}) if rollout_cache.history else {}
                    success = episode_metrics.get('success', False)
                    reward = episode_metrics.get('raw_reward', 0)
                steps = rollout_cache.step
                
                self.logger.info(f"[EPISODE END] Group:{self.env_config['group_id']} Env:{self.env_config['env_id']} "
                               f"Episode:{self.episode_id} Steps:{steps} Success:{success} Reward:{reward}")
                
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}
                
                rollout: DataProto = self.formulate_rollouts(rollout_cache, start_step)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                
                # 先mask掉非正常结束的数据
                if self.stop_reason != EpisodeStopReason.FINISH and self.stop_reason != EpisodeStopReason.MAX_LENGTH:
                    rollout.meta_info["drop_flag"] = True
                    
                if self.failure_mode == "reward_calculation_failed":
                    rollout.meta_info["drop_flag"] = True
                    
                if self.failure_mode == "ENV_TIMEOUT":
                    rollout.meta_info["drop_flag"] = True
                    
                
                self.logger.info(f"[DATA_SUBMIT] START - TrajID:{traj_id}")
                
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))
                
                self.logger.info(f"[DATA_SUBMIT] Success! - TrajID:{traj_id} submitted successfully")
                self.logger.info(f"MaxTraj:{self.worker_config.max_traj_per_env}")

                self.logger.info(f"[ROLLOUT_LOOP] Success! - Group:{self.env_config['group_id']} Env:{self.env_config['env_id']} Final Success:{success} Final Reward:{reward} Episodes:{self.episode_id}, Steps:{steps}, MaxTraj:{self.worker_config.max_traj_per_env}")
                
                self.env.close()
                time.sleep(300)
                rollout_cache = self.reset()
                start_step = self.current_step
        ray.get(self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, None))

    def reset(self) -> Optional[RolloutCache]:
        """Reset the environment and start a new episode"""
        self.logger.info(f"[ENV_RESET] START - Group:{self.env_config['group_id']} Env:{self.env_config['env_id']} Episode:{self.episode_id}")
        self.env_instance_start_time = datetime.now()
        self.failure_mode = None
        self.stop_reason = None
        self.reset_status = False
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                        group_id=self.env_config['group_id'],
                                        tag=self.env_config['tag'])
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config["group_id"]))
        if self.episode_id is None:
            assert not self.running
            self.logger.info(f"[ENV_END] group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id}")
            return None
        seed = self.group_seed + self.episode_id
        
        with self.thread_lock:
            observation, info = self.env.reset(seed=seed)
        
        if observation is None:
            self.logger.info(f"[ENV_END] group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id}")
            return None
        

        if self.env.env_reset_failed:
            self.logger.error(f"[ENV_RESET] Failed! - Environment reset failed, observation: {json.dumps(observation, ensure_ascii=False)}, env_reset_failed: {self.env.env_reset_failed}")
            self.reset_status = True
            self.failure_mode = info.get("failure_mode", "Sandbox Initialization Failed")
            self.stop_reason = EpisodeStopReason.ENV_RESET_FAILED
        else:
            self.logger.info(f"[ENV_RESET] Success! - Seed:{seed} TaskID:{getattr(self.env, 'task_id', 'N/A')}")
            self.reset_status = False
            
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,
            **info
        })
        return self.rollout_cache

    def step(self, llm_output: DataProto) -> Optional[RolloutCache]:
        """Execute a step in the environment based on the model's output"""
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)
        model_response = responses[0]
        
        step_num = self.rollout_cache.step + 1

        self.logger.info(f"[ENV_STEP] START - Group:{self.env_config['group_id']}, Env:{self.env_config['env_id']}, Episode:{self.episode_id}, Steps:{step_num}, Task_name: {self.env.task_name}, Sandbox_ip: {self.env.sandbox_util.sandbox_ip}, sandbox_id: {self.env.sandbox_util.sandbox_id}, Seed:{self.group_seed}, model_response: {json.dumps(model_response, ensure_ascii=False)}")
        
        with self.thread_lock:
            observation, reward, terminated, truncated, info = self.env.step(model_response)
        
        suffix = info.pop("suffix", None)
        action_valid = info.get('metrics', {}).get('action_is_valid', 0) if info else 0
        
        self.logger.info(f"[ENV_STEP] Success! - Group:{self.env_config['group_id']}, Env:{self.env_config['env_id']}, Episode:{self.episode_id}, Steps:{step_num}, Task_name: {self.env.task_name}, Step:{step_num}, Reward:{reward} Valid:{action_valid} Term:{terminated} Trunc:{truncated}, observation: {json.dumps(observation, ensure_ascii=False)}")
        
        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            self.rollout_cache.truncated = True
            self.stop_reason = EpisodeStopReason.MAX_STEPS
                

        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = model_response
        
        if info is not None:
            self.rollout_cache.history[-1].update(info)
            self.failure_mode = info.get("failure_mode", "")
            self.rollout_cache.history[-1]['use_tool'] = info.get('metrics', {}).get('action_is_valid', 0) if info else 0
            
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        })

        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix
            

        if self.mode == "val" and self.pipeline_config.render_save_dir and hasattr(self.env, "render"):
            frame = self.env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache) -> DataProto:
        """
        Make a decision using the roll framework for model generation and iflow-cli for tool execution.
        """
        lm_input, input_messages = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]
        
        # if random.randint(0, 50) < 1:
        #     input_string = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        #     self.logger.info(f"!!input_string: {input_string}")
        if input_messages[0]['role'] == "system" and len(input_messages) < 3:
            self.logger.warning(f"input_messages len:{len(input_messages)} < 3, get system prompt failed")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.NO_SYSTEM_PROMPT})
        
            
        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]}, "
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        input_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        self.logger.info(f"[LLM_GENERATE] START - input_messages len:{len(input_messages)}")
        try:
            lm_output: DataProto = self.llm_proxy.generate(messages=input_messages,
                                                           lm_input=lm_input,
                                                           generation_config=generation_config)
            if lm_output is None:
                self.logger.error(f"[LLM_GENERATE] Failed! - LLM returned None output")
                self.failure_mode = "llm_generate_failed"
                return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
            
            self.logger.info(f"[LLM_GENERATE] Success! - lm_output len:{len(lm_output)}")
        except Exception as e:
            self.stop_reason = EpisodeStopReason.LLM_GENERATE_FAILED
            self.failure_mode = "llm_generate_failed"
            self.logger.error(f"[LLM_GENERATE] Failed! - Error: {str(e)}")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()
        content = self.rollout_cache.history[-1]
        content["response_ids"] = response_ids
        content["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output
    
    def final_system_prompt(self, messages, tokenizer, tools = None):
        if isinstance(messages, str):
            messages = json.loads(messages)
        if  messages[0]['role'] == "system":
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False
            )
            system_prompt = full_prompt.split("<|im_end|>\n")[0].split("<|im_start|>system\n")[-1].strip()
            messages[0]['content'] = system_prompt
        return messages
    
    def post_messages(self, messages, tokenizer, tools = None):
        if isinstance(messages, str):
            messages = json.loads(messages)
        messages_new = []
        try:
            for mess in messages:
                if isinstance(mess, str):
                    try:
                        mess = json.loads(mess)
                        content = mess.get("content", "")
                    except Exception as e:
                        self.logger.info(f"Error calling iflow-cli post_messages: {e}, mess: {mess}")
                        
                        self.logger.info(f"traceback: {traceback.format_exc()}")
                        content = str(mess)
                else:
                    content = mess.get("content", "")
                if isinstance(content, list):
                    all_content = []
                    for cc in content:
                        if isinstance(cc, dict):
                            all_content.append(cc.get("text", ""))
                        else:
                            all_content.append(str(cc))
                    content = "\n".join(all_content)
                else:
                    content = str(content)
                mess['content'] = content
                messages_new.append(mess)
        except Exception as e:
            self.logger.info(f"Error calling iflow-cli post_messages: {e}")
            self.logger.info(f"traceback: {traceback.format_exc()}")
        return messages_new

    def format_messages(self, history: RolloutCache):
        """
        Format the messages for the model input using iflow-cli framework.
        """
        content = self.rollout_cache.history[-1]
        
        question = ""
        messages = []
        if  self.env.tools:
            tools = self.env.tools
        else:
            tools = None
            
        if self.rollout_cache.step == 0:
            question = f"{content['observation']}"
            try:
                self.logger.info(f"[GET_SYSINFO] START - question:{question[:100]}")
                messages, error_info = self.env.get_sysinfo(question)
                if error_info:
                    self.logger.error(f"[GET_SYSINFO] Failed! - iflow-cli get_messages error: {error_info}")
                    messages = [{"role": "user", "content": question}]
                else:
                    self.logger.info(f"[GET_SYSINFO] Success! - messages len:{len(messages)}")
                    # if random.randint(0, 50) < 1:
                    #     self.logger.info(f"iflow-cli messages: {messages}")
                    messages = self.post_messages(messages, self.tokenizer, tools)
                    messages = self.final_system_prompt(messages, self.tokenizer, tools)
            except Exception as e:
                self.logger.error(f"[GET_SYSINFO] Failed! - Error calling iflow-cli get_messages: {e}")
                self.logger.warning(f"[GET_SYSINFO] Using fallback messages due to exception")
                messages = [{"role": "user", "content": question}]
        else:
            # render_dict = {"observation": content["observation"]}
            # if contains_renderable_field(self.agent_template, "turn_idx"):
            #     render_dict["turn_idx"] = self.rollout_cache.step + 1
            # if contains_renderable_field(self.agent_template, "actions_left"):
            #     render_dict["actions_left"] = content["actions_left"]
            # if contains_renderable_field(self.agent_template, "max_response_length"):
            #     render_dict["max_response_length"] = self.env_config.get("max_tokens_per_step", 2048)
            question = content["observation"]
            messages.append({"role": "user", "content": question})
            
        content["messages"] = messages
        messages = self.post_messages(messages, self.tokenizer, tools)


        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        
        if len(self.rollout_cache.history) > 1 and "response_ids" in self.rollout_cache.history[-2]:
            prev_response_ids = self.rollout_cache.history[-2]["response_ids"]
            prev_response = self.tokenizer.decode(prev_response_ids, skip_special_tokens=False)
            
            if not prev_response.endswith("<|im_end|>\n"):
                im_end_token_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                if prev_response.endswith("<|im_end|>"):
                    newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)
                    prompt_ids = newline_token_id + prompt_ids
                else:
                    prompt_ids = im_end_token_ids + prompt_ids
        
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            if "prompt_ids" in items and "response_ids" in items:
                history_token_ids.extend(items["prompt_ids"])
                history_token_ids.extend(items["response_ids"])
        input_ids = history_token_ids + prompt_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        
        content["prompt_ids"] = prompt_ids
        content["messages"] = messages
        return lm_input, messages

    def formulate_rollouts(self, rollout_cache: RolloutCache, start_step: int):
        last_observation = ""
        if 'observation' in rollout_cache.history[-1]:
            last_observation = rollout_cache.history[-1]['observation']
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)
        token_ids = []
        prompt_masks = []
        response_masks = []
        
        for items in self.rollout_cache.history:
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))

        input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        first_response_idx = response_masks.index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask =torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        first_round_reward = 0
        if len(scores) > 0:
            first_round_reward = scores[0]

        traj_rollout_time = float(getattr(self.env, 'traj_rollout_time', 0.0))
        traj_env_time = float(getattr(self.env, 'traj_env_time', 0.0))
        env_timeout = float(getattr(self.env, 'env_timeout', 0.0))
        env_reset_failed = float(getattr(self.env, 'env_reset_failed', 0.0))

        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
            "first_round_reward": np.array([first_round_reward], dtype=object),
            "traj_rollout_time": np.array([traj_rollout_time], dtype=object),
            "traj_env_time": np.array([traj_env_time], dtype=object),
            "env_timeout": np.array([env_timeout], dtype=object),
            "env_reset_failed": np.array([env_reset_failed], dtype=object),
        })
        env_metric = {
            "traj_rollout_time": traj_rollout_time,
            "traj_env_time": traj_env_time,
            "env_timeout": env_timeout,
            "env_reset_failed": env_reset_failed,
            "episode_score": episode_score
        }
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}
        
        
        self.env_instance_end_time = datetime.now()
        trajectory_data = self._save_trajectory_to_file(rollout_cache, scores, episode_score, response_length, env_metric, last_observation, start_step)
        
        lm_input.non_tensor_batch["trajectory_data"] = np.array([json.dumps(trajectory_data)], dtype=object)
        colummns_config = [
            ["trajectory_data", "string"]
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        
        return lm_input
    
    def _save_trajectory_to_file(self, rollout_cache: RolloutCache, scores: list, episode_score: float, response_length: float, env_metric: dict, last_observation: str = "", start_step: int = -1):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            task_name = getattr(self.env, 'task_name', 'N/A')
            traj_id = f"{rollout_cache.tag}_{start_step}_{rollout_cache.group_id}_{rollout_cache.env_id}_{self.episode_id}_{self.group_seed}_{task_name}_{timestamp}"
            
            all_messages = [item for items in self.rollout_cache.history for item in items["messages"]]
        
            last_step_info = rollout_cache.history[-1] if rollout_cache.history else {}
            failure_mode = last_step_info.get('failure_mode', '')
            stop_reason = self.stop_reason.value if self.stop_reason else last_step_info.get('stop_reason', '')
            error_messages = last_step_info.get('error_messages', [])
            test_output = last_step_info.get('test_output', '')
        
            env_instance_duration = None
            if self.env_instance_start_time and self.env_instance_end_time:
                env_instance_duration = (self.env_instance_end_time - self.env_instance_start_time).total_seconds()

            trajectory_data = {
                "trajectory_id": traj_id,
                "timestamp": datetime.now().isoformat(),
                "currect_step": self.current_step,
                "env_info": {
                    "env_id": rollout_cache.env_id,
                    "group_id": rollout_cache.group_id,
                    "tag": rollout_cache.tag,
                    "task_name": getattr(self.env, 'task_name', 'N/A'),
                    "task_id": getattr(self.env, 'task_id', 'N/A'),
                    "sandbox_image": getattr(self.env, 'sandbox_image', 'N/A'),
                    "sandbox_ip": getattr(self.env.sandbox_util, 'sandbox_ip', 'N/A') if hasattr(self.env, 'sandbox_util') else 'N/A',
                    "sandbox_id": getattr(self.env.sandbox_util, 'sandbox_id', 'N/A') if hasattr(self.env, 'sandbox_util') else 'N/A',
                    "seed": self.group_seed,
                    "episode_id": self.episode_id,
                    "max_steps": self.env_config.max_steps,
                    "mode": self.mode
                },
                "timing_info": {
                    "env_instance_start_time": self.env_instance_start_time.isoformat() if self.env_instance_start_time else None,
                    "env_instance_end_time": self.env_instance_end_time.isoformat() if self.env_instance_end_time else None,
                    "env_instance_duration_seconds": env_instance_duration,
                    "trajectory_save_time": datetime.now().isoformat()
                },

                "length_info": {
                    "trajectory_length": rollout_cache.step,
                    "num_actions": rollout_cache.step,
                    "response_length": response_length,
                    "total_tokens": len([item for items in rollout_cache.history for item in (items.get("prompt_ids", []) + items.get("response_ids", []))]),
                    "terminated": rollout_cache.terminated,
                    "truncated": rollout_cache.truncated
                },
                "reward_info": {
                    "episode_score": episode_score,
                    "step_scores": scores,
                    "first_round_reward": scores[0] if scores else 0,
                    "final_reward": scores[-1] if scores else 0
                },
                "failure_info": {
                    "failure_mode": failure_mode,
                    "stop_reason": stop_reason,
                    "error_messages": error_messages,
                    "test_output": test_output,
                    "has_failure": bool(failure_mode and failure_mode not in ['', 'none']),
                    "failure_step": None
                },
                "metrics": env_metric,
                "messages": all_messages,
                "last_observation": last_observation
            }
            
            filename = f"trajectory_{traj_id}.json"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"[TRAJ_SAVE] Success! - Trajectory saved to: {filepath}")
            self.logger.info(f"[TRAJ_SAVE] Summary - TrajID:{traj_id}, Steps:{rollout_cache.step}, Score:{episode_score}, Success:{env_metric.get('env/terminal_bench/success', False)}, FailureMode:{failure_mode}, StopReason:{stop_reason}")
            return trajectory_data
        except Exception as e:
            self.logger.error(f"[TRAJ_SAVE] Failed! - Error saving trajectory: {str(e)}")
            self.logger.error(f"[TRAJ_SAVE] Traceback: {traceback.format_exc()}")
            return {}

    def _create_placeholder_rollout(self, episode_id: int, failure_reason: str) -> DataProto:
        """
        Create a minimal placeholder rollout with response_mask=1 to skip loss calculation.
        """
        self.logger.info(f"[PLACEHOLDER_ROLLOUT] Failure reason: {failure_reason}")
        seq_len = 10
        
        input_ids = torch.full((1, seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        position_ids = torch.zeros((1, seq_len), dtype=torch.long)
        response_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        prompt_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        score_tensor = torch.zeros((1, seq_len), dtype=torch.float)
        
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        }, batch_size=1)

        lm_input.non_tensor_batch = {
            "env_ids": np.array([self.env_config['env_id']], dtype=object),
            "group_ids": np.array([self.env_config['group_id']], dtype=object),
            "tags": np.array([self.env_config['tag']], dtype=object),
            "frames": np.array([[]], dtype=object),
            "step_scores": np.array([[0]], dtype=object),
            "episode_scores": np.array([0], dtype=object),
            "first_round_reward": np.array([0], dtype=object),
            "traj_rollout_time": np.array([0.0], dtype=object),
            "traj_env_time": np.array([0.0], dtype=object),
            "env_timeout": np.array([1.0], dtype=object),
            "env_reset_failed": np.array([1.0], dtype=object)
        }
        
        traj_group_id = f"{self.env_config['tag']}_{self.env_config['group_id']}_{episode_id}_{self.group_seed}"
        traj_id = f"{traj_group_id}_{self.env_config['env_id']}"
        lm_input.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * lm_input.batch.batch_size[0], dtype=object)
        lm_input.non_tensor_batch["traj_id"] = np.array([traj_id] * lm_input.batch.batch_size[0], dtype=object)

        env_metric = {
            "num_actions": 0,
            "success": 0,
        }
        env_metric = {f"env/{self.env_config['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = 0.0
        lm_input.meta_info = {"metrics": env_metric}
        
        
        lm_input.non_tensor_batch["trajectory_data"] = np.array([""], dtype=object)
        colummns_config = [
            ["trajectory_data", "string"]
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        
        return lm_input


class GroupFilterTB:
    def __init__(self, config: AgenticConfig, env_manager_config: EnvManagerConfig, mode: str):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode
        self.global_filter_stats = {"total": 0, "filtered": 0}

    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        self.global_filter_stats["total"] += 1
        should_drop = False
        for data in group:
            if data.meta_info.get("drop_flag", False):
                should_drop = True
                break
        
        if not should_drop:
            return False
        
        current_global_filter_ratio = (
            self.global_filter_stats["filtered"] / self.global_filter_stats["total"] 
            if self.global_filter_stats["total"] > 0 else 0.0
        )
        
        if current_global_filter_ratio >= 0.5:
            return False
        
        if (self.global_filter_stats["filtered"] + 1) / self.global_filter_stats["total"] > 0.5:
            return False
        
        self.global_filter_stats["filtered"] += 1
        return True
