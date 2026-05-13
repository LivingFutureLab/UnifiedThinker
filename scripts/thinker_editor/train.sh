#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

set -x

CFG=config/model/thinker_editor/notebook.yaml
GPU=4         # GPU

# set deepspeed config file based on stage
DEEPSPEED_CONFIG_FILE="config/deepspeed/deepspeed_config_zero2.yaml"

# 构建运行参数
args="--config_file=${DEEPSPEED_CONFIG_FILE} train_thinker_editor.py -c ${CFG} -z zero2"
echo $args

#CUDA_VISIBLE_DEVICES="0" 
accelerate launch --num_processes=${GPU} ${args}

