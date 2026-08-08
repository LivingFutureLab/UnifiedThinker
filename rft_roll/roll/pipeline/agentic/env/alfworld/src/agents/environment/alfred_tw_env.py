import os
import json
import random
import numpy as np

from tqdm import tqdm
from termcolor import colored

import textworld
import textworld.agents
import textworld.gym

from roll.pipeline.agentic.env.alfworld.src.agents.utils.misc import Demangler, add_task_to_grammar
from roll.pipeline.agentic.env.alfworld.src.agents.expert import HandCodedTWAgent, HandCodedAgentTimeout


TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._entity_infos, shuffle=self.shuffle)
        for info in self._entity_infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


class AlfredInfos(textworld.core.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        state["extra.gamefile"] = self._gamefile
        return state


# Enum for the supported types of AlfredExpert.
class AlfredExpertType:
    HANDCODED = "handcoded"
    PLANNER = "planner"


class AlfredExpert(textworld.core.Wrapper):

    def __init__(self, env=None, expert_type=AlfredExpertType.HANDCODED):
        super().__init__(env=env)

        self.expert_type = expert_type
        self.prev_command = ""

        if expert_type not in (AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER):
            msg = "Unknown type of AlfredExpert: {}.\nExpecting either '{}' or '{}'."
            msg = msg.format(expert_type, AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER)
            raise ValueError(msg)

    def _gather_infos(self):
        # Compute expert plan.
        if self.expert_type == AlfredExpertType.HANDCODED:
            self.state["extra.expert_plan"] = ["look"]
            try:
                # initialization
                if not self.prev_command:
                    self._handcoded_expert.observe(self.state["feedback"])
                else:
                    handcoded_expert_next_action = self._handcoded_expert.act(self.state, 0, self.state["won"], self.prev_command)
                    if handcoded_expert_next_action in self.state["admissible_commands"]:
                        self.state["extra.expert_plan"] = [handcoded_expert_next_action]
            except HandCodedAgentTimeout:
                raise Exception("Timeout")
        elif self.expert_type == AlfredExpertType.PLANNER:
            self.state["extra.expert_plan"] = self.state["policy_commands"]
        else:
            raise NotImplementedError("Unknown type of AlfredExpert: {}.".format(self.expert_type))

    def load(self, gamefile):
        super().load(gamefile)
        self.gamefile = gamefile
        self.request_infos.policy_commands = self.request_infos.policy_commands or (self.expert_type == AlfredExpertType.PLANNER)
        self.request_infos.facts = self.request_infos.facts or (self.expert_type == AlfredExpertType.HANDCODED)
        self._handcoded_expert = HandCodedTWAgent(max_steps=200)

    def step(self, command):
        self.state, reward, done = super().step(command)
        self.prev_command = str(command)
        self._gather_infos()
        return self.state, reward, done

    def reset(self):
        self.state = super().reset()
        self._handcoded_expert.reset(self.gamefile)
        self.prev_command = ""
        self._gather_infos()
        return self.state


class AlfredTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, input_game_files, train_eval="train"):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval
        # self.game_idx = game_idx
        # self.game_files = input_game_files.sort()
        self.eval_game = input_game_files
        if config["env"]["goal_desc_human_anns_prob"] > 0:
            msg = ("Warning! Changing `goal_desc_human_anns_prob` should be done with"
                   " the script `alfworld-generate`. Ignoring it and loading games as they are.")
            print(colored(msg, "yellow"))

    def get_game_logic(self):
        self.game_logic = {
            "pddl_domain": open(os.path.expandvars(self.config['logic']['domain'])).read(),
            "grammar": open(os.path.expandvars(self.config['logic']['grammar'])).read()
        }

    # use expert to check the game is solvable
    def is_solvable(self, env, game_file_path,
                    random_perturb=True, random_start=10, random_prob_after_state=0.15):
        done = False
        steps = 0
        trajectory = []
        try:
            env.load(game_file_path)
            game_state = env.reset()
            if env.expert_type == AlfredExpertType.PLANNER:
                return game_state["extra.expert_plan"]

            while not done:
                expert_action = game_state['extra.expert_plan'][0]
                random_action = random.choice(game_state.admissible_commands)

                command = expert_action
                if random_perturb:
                    if steps <= random_start or random.random() < random_prob_after_state:
                        command = random_action

                game_state, _, done = env.step(command)
                trajectory.append(command)
                steps += 1
        except Exception as e:
            print("Unsolvable: %s (%s)" % (str(e), game_file_path))
            return None

        return trajectory

    def init_env(self, batch_size):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        # Register a new Gym environment.
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        # print(request_infos)
        expert_type = self.config["env"]["expert_type"]
        training_method = self.config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]

            expert_plan = True if self.train_eval == "train" else False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")

        else:
            raise NotImplementedError

        # print(list(self.eval_game))
        # print(self.eval_game)

        env_id = textworld.gym.register_game(self.eval_game, request_infos,
                                              batch_size=batch_size,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode,
                                              wrappers=wrappers)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env
