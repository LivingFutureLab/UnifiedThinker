import requests
import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable, Sized
import time
import re
from termcolor import colored
from utils import *

subtask_dic = {
    "Temp": [
        "Life Progression",
        "Material Progression",
        "Environmental Cycles",
        "Societal Transformation",
    ],
    "Causal": [
        "Structural Deformation",
        "State Transition",
        "Chemical and Biological Transformation",
        "Physics Manifestation",
    ],
    "Spa": [
        "Component Assembly",
        "Object Arrangement",
        "Viewpoint Generation",
        "Structural Inference",
        "Layout Reasoning",
    ],
    "Logic": ["Pattern Prediction", "Mathematical Derivation", "Puzzle Solving"],
}

def extract(answer):
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return numbers
    else:
        return None

def calculate_score(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        if 'consistency_free' in row and row['consistency_free']:
            score = 0.2 * row['VisualPlausibility'] + 0.8 * row['Reasoning']
        else:
            score = 0.3 * row['ApprConsistency'] + 0.5 * row['Reasoning'] + 0.2 * row['VisualPlausibility']
        
    elif row['category'] == 'logical_reasoning':
        score = 0.3 * row['ApprConsistency'] + 0.7 * row['Reasoning']
    if row['Reasoning'] == 1:
        score = score * 0.5
        score = 1 if score<1 else score
    return score

def calculate_completion(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        return (
            1
            if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 and row['VisualPlausibility'] == 5
            else 0
        )
    elif row['category']=='logical_reasoning':
        return (
            1 if row['ApprConsistency'] == 5 and row['Reasoning'] == 5 else 0
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/data/oss_bucket_0/jianchong.zq/benchmarks/RISEBench/datav2_total_w_subtask_only_temp_cuasal.json")    
    parser.add_argument('-r', '--pkl_result_file', type=str)

    args = parser.parse_args()


    data = json.load(open(args.data))
    data = pd.DataFrame(data)

    result = load(args.pkl_result_file)

    judges = [result[i] for i in data['index']]

    scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []

    for judge in judges:
        if judge['judge1'] is None:
            judge_combine.append(
                'REASONING\n\n'
                + judge['judge2']
                + '\n\nQUALITY\n\n'
                + judge['judge3']
            )
            judge_cons.append(None)
            judge_reas.append(judge['judge2'])
            judge_qua.append(judge['judge3'])

            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score2 or not score3:
                score=None
            else:
                score = [None]+score2+score3
        elif 'judge3' not in judge:
            judge_combine.append(
                'CONSISTENCY\n\n'
                + judge['judge1']
                + '\n\nREASONING\n\n'
                + judge['judge2']
            )
            judge_cons.append(judge['judge1'])
            judge_reas.append(judge['judge2'])
            judge_qua.append(None)

            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            if not score1 or not score2:
                score=None
            else:
                score = score1+score2
        elif 'judge2' not in judge:
            judge_combine.append(judge['judge1'])
            score = [extract(judge['judge1'])[1], extract(judge['judge1'])[0]]
        else:
            try:
                judge_combine.append(
                    'CONSISTENCY\n\n'
                    + judge['judge1']
                    + '\n\nREASONING\n\n'
                    + judge['judge2']
                    + '\n\nQUALITY\n\n'
                    + judge['judge3']
                )
                judge_cons.append(judge['judge1'])
                judge_reas.append(judge['judge2'])
                judge_qua.append(judge['judge3'])
            except Exception as e:
                print(e)
                breakpoint()
            score1 = extract(judge['judge1'])
            score2 = extract(judge['judge2'])
            score3 = extract(judge['judge3'])
            if not score1 or not score2 or not score3:
                score=None
            else:
                score = score1+score2+score3
        scores.append(score)

    reasoning = []
    img_consist = []
    gen_quality = []
    match_log = []

    for score in scores:
        if score:
            match_log.append('succeed')
            if len(score)==3:
                img_consist.append(score[0])
                reasoning.append(score[1])
                gen_quality.append(score[2])

            elif len(score)==2:
                reasoning.append(4 * min(score[1], 1) + 1)
                img_consist.append(4 * min(score[0], 1) + 1)
                gen_quality.append(None)
        else:
            img_consist.append(None)
            reasoning.append(None)
            gen_quality.append(None)
            match_log.append('failed')
    # breakpoint()
    data['Reasoning'] = reasoning
    data['ApprConsistency'] = img_consist
    data['VisualPlausibility'] = gen_quality
    data['match_log'] = match_log
    
    print(colored(f"match_log: {len(match_log)} samples; 其中 {sum([e == 'failed' for e in match_log])} failed samples.", "green", attrs=["bold"]))
    
    # data['judge'] = judge_combine
    data['judge_cons'] = judge_cons
    data['judge_reas'] = judge_reas
    data['judge_qua'] = judge_qua

    data['score'] = data.apply(calculate_score, axis=1)
    data['complete'] = data.apply(calculate_completion, axis=1)

    #dump(data, judge_res)

    df_causal = data[data['category'] == 'causal_reasoning']
    df_temporal = data[data['category'] == 'temporal_reasoning']
    df_spatial = data[data['category'] == 'spatial_reasoning']
    df_logical = data[data['category'] == 'logical_reasoning']

    score_final = data['score'].mean()
    completion_rate = data['complete'].mean()
    
    # calculate score and accuracy per main task
    temporal_final, temporal_comp_rate = df_temporal['score'].mean(), df_temporal['complete'].mean()
    causal_final, causal_comp_rate = df_causal['score'].mean(), df_causal['complete'].mean()
    spatial_final, spatial_comp_rate = df_spatial['score'].mean(), df_spatial['complete'].mean()
    logical_final, logical_comp_rate = df_logical['score'].mean(), df_logical['complete'].mean()

    reasoning_average = data['Reasoning'].mean()
    img_consist_average = data['ApprConsistency'].mean()
    generation_quality = data['VisualPlausibility'].mean()

    temp_rea_avg, temp_cons_avg, temp_qua_avg = df_temporal['Reasoning'].mean(), df_temporal['ApprConsistency'].mean(), df_temporal['VisualPlausibility'].mean()
    cau_rea_avg, cau_cons_avg, cau_qua_avg = df_causal['Reasoning'].mean(), df_causal['ApprConsistency'].mean(), df_causal['VisualPlausibility'].mean()
    spa_rea_avg, spa_cons_avg, spa_qua_avg = df_spatial['Reasoning'].mean(), df_spatial['ApprConsistency'].mean(), df_spatial['VisualPlausibility'].mean()
    logic_rea_avg, logic_cons_avg, logic_qua_avg = df_logical['Reasoning'].mean(), df_logical['ApprConsistency'].mean(), df_logical['VisualPlausibility'].mean()

    def trans_to_percent(s):
        return 25*(s-1)
    
    # calculate score and accuracy per subtask
    average_scores_by_subtask = data.groupby('subtask')['score'].mean()
    average_acc_by_subtask = data.groupby('subtask')['complete'].mean()

    average_scores_dict = average_scores_by_subtask.to_dict()
    average_acc_dict = average_acc_by_subtask.to_dict()
    
    subtask_results = {}
    for k, v in average_scores_dict.items():
        subtask_results[k] = [v, trans_to_percent(v), average_acc_dict[k]]
    
    sorted_subtask_results = {}
    for main_task_prefix, subtasks in subtask_dic.items():
        for subtask in subtasks:
            if subtask in subtask_results:
                new_key = f"{main_task_prefix}-{subtask}"
                sorted_subtask_results[new_key] = subtask_results[subtask]

    final_score = dict(
        Overall=[score_final, trans_to_percent(score_final), completion_rate],
        Temporal=[temporal_final, trans_to_percent(temporal_final), temporal_comp_rate],
        Causal=[causal_final, trans_to_percent(causal_final), causal_comp_rate],
        Spatial=[spatial_final, trans_to_percent(spatial_final), spatial_comp_rate],
        Logical=[logical_final, trans_to_percent(logical_final), logical_comp_rate],
        Overall_Reasoning=[reasoning_average, trans_to_percent(reasoning_average), None],
        Overall_ApprConsistency=[img_consist_average, trans_to_percent(img_consist_average), None],
        Overall_VisualPlausibility_total=[generation_quality, trans_to_percent(generation_quality), None],
        Temporal_Reasoning = [temp_rea_avg, trans_to_percent(temp_rea_avg), None],
        Temporal_Consistency = [temp_cons_avg, trans_to_percent(temp_cons_avg), None],
        Temporal_Quality = [temp_qua_avg, trans_to_percent(temp_qua_avg), None],
        Causal_Reasoning = [cau_rea_avg, trans_to_percent(cau_rea_avg), None],
        Causal_Consistency = [cau_cons_avg, trans_to_percent(cau_cons_avg), None],
        Causal_Quality = [cau_qua_avg, trans_to_percent(cau_qua_avg), None],
        Spatial_Reasoning = [spa_rea_avg, trans_to_percent(spa_rea_avg), None],
        Spatial_Consistency = [spa_cons_avg, trans_to_percent(spa_cons_avg), None],
        Spatial_Quality = [spa_qua_avg, trans_to_percent(spa_qua_avg), None],
        Logical_Reasoning = [logic_rea_avg, trans_to_percent(logic_rea_avg), None],
        Logical_Consistency = [logic_cons_avg, trans_to_percent(logic_cons_avg), None],
        **sorted_subtask_results
    )

    print("-" * 100)
    print(final_score)
    # df = pd.DataFrame(final_score, index=["Score-Origin", "Score-Percentage", "Accuracy"]).T
    # df.reset_index(inplace=True)
    # df.columns = ["-", "Score-Origin", "Score-Percentage", "Accuracy"]
    # df.to_csv(score_file, index=False)
    # print(f"write to {score_file}")


if __name__ == '__main__':
    main()
