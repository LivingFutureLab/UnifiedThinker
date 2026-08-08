import os
import json
import random
import numpy as np
import yaml
from tqdm import tqdm
# from termcolor import colored
import jsonlines

TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}



def collect_game_files(train_eval, env_config, data_path, labeled_data, verbose=False):
    def log(info):
        if verbose:
            print(info)
    
    with open(env_config) as reader:
        config = yaml.safe_load(reader)

    game_files = []
    
    log("Collecting solvable games...")
    # print(data_path)
    # get task types
    assert len(config['env']['task_types']) > 0
    task_types = []
    for tt_id in config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_types.append(TASK_TYPES[tt_id])
    # print(labeled_data)
    count = 0
    # print(data_path)
    for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
     
        if root not in labeled_data:
            # print(1)
            continue

        if 'traj_data.json' in files:
            count += 1
            # print(1)
            # Filenames
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")
          
            if 'movable' in root or 'Sliced' in root:
                log("Movable & slice trajs not supported %s" % (root))
                continue

            # Get goal description
            with open(json_path, 'r') as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            if not traj_data['task_type'] in task_types:
                log("Skipping task type")
                continue

            # Check if a game file exists
            if not os.path.exists(game_file_path):
                log(f"Skipping missing game! {game_file_path}")
                continue

            with open(game_file_path, 'r') as f:
                gamedata = json.load(f)

            # Check if previously checked if solvable
            if 'solvable' not in gamedata:
                print(f"-> Skipping missing solvable key! {game_file_path}")
                continue

            if not gamedata['solvable']:
                log("Skipping known %s, unsolvable game!" % game_file_path)
                continue

            # Add to game file list
            game_files.append(game_file_path)

    print(f"Overall we have {len(game_files)} games in split={train_eval}")
    num_games = len(game_files)

    if train_eval == "train":
        # num_train_games = config['dataset']['num_train_games'] if config['dataset']['num_train_games'] > 0 else len(game_files)
        # game_files = game_files[:num_train_games]
        # num_games = len(game_files)
        print("Training with %d games" % (len(game_files)))
    else:
        # num_eval_games = config['dataset']['num_eval_games'] if config['dataset']['num_eval_games'] > 0 else len(game_files)
        # game_files = game_files[:num_eval_games]
        # num_games = len(game_files)
        print("Evaluating with %d games" % (len(game_files)))
    
    return sorted(game_files)

