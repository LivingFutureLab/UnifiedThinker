import os 
import re
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def plot_log(log_file, score_keys):
    global_steps = []
    #mean_scores = []
    mean_scores = {k: [] for k in score_keys}
    time_step_generate = []
    response_length = []
    
    step_time_start = []
    step_time_finish = []
    
    num_plots = len(score_keys) + 1
    
    with open(log_file, 'r') as f:
        for line in f.readlines():
            if "system/step" in line and f"{score_keys[0]}/critic/score/mean" in line:
                match = re.search(r'"system/step":\s*(\d+)', line)
                step = int(match.group(1))
                global_steps.append(step)
                
                # match = re.search(r'"reason_edit/critic/score/mean":\s*([\d.]+)', line)
                # score = float(match.group(1))
                # mean_scores.append(score)
                for k in score_keys:
                    match = re.search(rf'"{k}/critic/score/mean":\s*([\d.]+)', line)
                    score = float(match.group(1))
                    mean_scores[k].append(score)
                    
                
                match = re.search(r'"time/step_generate":\s*([\d.]+)', line)
                t = float(match.group(1))
                time_step_generate.append(t)
                
                match = re.search(r'"token/response_length/mean":\s*([\d.]+)', line)
                length = float(match.group(1))
                response_length.append(length)
            
            if "pipeline step " in line and 'finished' in line:
                try:
                    # e.g., [2025-11-07 19:43:16] [rlvr_image_think_pipeline_v1.py (933)] [INFO] [DRIVER 0 / 8][PID 1346] pipeline step 9 finished
                    match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?pipeline step (\d+) finished', line)
                    time_str = match.group(1)
                    dt_object = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    timestamp_seconds = dt_object.timestamp()
                    step = int(match.group(2))
                    step_time_finish.append(timestamp_seconds)
                    assert len(step_time_finish) == step + 1
                except Exception as e:
                    print("error of {}".format(str(e)))
                    print("line: {}".format(line))
            
            if "pipeline step " in line and 'start' in line:
                try:
                    # e.g., [2025-11-07 19:43:16] [rlvr_image_think_pipeline_v1.py (933)] [INFO] [DRIVER 0 / 8][PID 1346] pipeline step 9 start
                    match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?pipeline step (\d+) start', line)
                    time_str = match.group(1)
                    dt_object = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    timestamp_seconds = dt_object.timestamp()
                    step = int(match.group(2))
                    step_time_start.append(timestamp_seconds)
                    assert len(step_time_start) == step + 1, "{} != {}".format(len(step_time_start), step)
                except Exception as e:
                    print("error of {}".format(str(e)))
                    print("line: {}".format(line))
    
    step_time = []
    for s, e in zip(step_time_start, step_time_finish):
        step_time.append(e - s)     
    step_time_avg = np.mean(step_time)
                
    print("average time_step_generate: {:.1f}".format(np.mean(time_step_generate)))
    print("average step time total: {:.1f}".format(step_time_avg))
    
    for i in range(len(score_keys)):
        plt.subplot(num_plots, 1, i+1)
        # for k in mean_scores:
        #     plt.plot(global_steps, mean_scores[k], '--*', label=k)
        k = score_keys[i]
        plt.plot(global_steps, mean_scores[k], '--*', label=k)
        
        # 适用于观察更复杂的波动趋势，通常 3 或 5 比较合适
        degree = 3
        z = np.polyfit(global_steps, mean_scores[k], degree)
        p = np.poly1d(z)
        plt.plot(global_steps, p(global_steps), linestyle='-', linewidth=2)
        
        #plt.xlabel("step")
        plt.ylabel("reward")
        plt.legend()
        plt.grid()
    
    plt.subplot(num_plots, 1, num_plots)
    plt.plot(global_steps, response_length, '--*')
    plt.xlabel("step")
    plt.ylabel("response_length")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--log_file', type=str,nargs="+")
    args = parser.parse_args()
    
    assert len(args.log_file) <= 2
    
    #score_keys = ['reason_edit', 'general_edit', 't2i']
    #score_keys = ['reason_edit', 'reason_t2i', 'general_t2i', 'general_edit']
    #score_keys = ['t2i']
    score_keys = ['reason_edit', 'reason_t2i', 'general_edit']
    #score_keys = ['general_edit']
    
    for log_file in args.log_file:
        plot_log(log_file, score_keys)
    
    plt.show()
    
    
    # step = [0, 480]
    # score = [3.31, 3.31]
    # plt.plot(step, score, 'b--', label="Qwen-Image-Edit-2509")
    
    # step2 = [0, 480]
    # score2 = [3.72, 3.72]
    # plt.plot(step2, score2, 'g--', label="Qwen-Image-Edit-2509 + zero-thinker")

    # steps_2 = [80, 160, 240, 320, 360, 400, 480]
    # scores_2 = [3.97, 4.05, 4.19 ,4.08, 4.21, 4.12, 4.19]
    
    
    # steps_1 = [80, 160, 240, 320, 400, 480]
    # scores_1 = [3.88, 3.97, 4.01, 4.10, 4.11, 4.12]
    
    # steps_3 = [80, 160, 240, 320, 400, 480]
    # scores_3 = [4.02, 4.16, 4.22, 4.24, 4.19, 4.24]
    
    # steps_4 = [160, 240, 320, 400, 480]
    # scores_4 = [4.26, 4.23, 4.24, 4.30, 4.34]
    
    # plt.plot(steps_1, scores_1, '*--', label="RFT: G=8")
    # plt.plot(steps_2, scores_2, '^--', label="RFT: G=14")
    # plt.plot(steps_3, scores_3, 'o--', label="RFT: G=28")
    # plt.plot(steps_4, scores_4, 'o--', label="RFT: G=42")
    # plt.legend()
    # plt.xlabel('step')
    # plt.ylabel("avg score")
    # #plt.ylim(0, 5)
    # plt.show()