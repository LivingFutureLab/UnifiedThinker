#!/bin/bash

# 启用命令回显
set -x

# 解析命令行参数
DEEPSPEED_STAGE=zero2
#DEEPSPEED_STAGE=zero3

CFG=config/model/qwen_image_edit_think/qwen_image_edit_think_notebook_${DEEPSPEED_STAGE}.yaml
GPU=1          # GPU数量

# 获取batch size
if ! BATCH="$(shyaml get-value train_bs < "${CFG}")"; then
    echo "Error: Failed to read lake value from config file"
    exit 1
fi
# 获取lr
if ! LR="$(shyaml get-value lr < "${CFG}")"; then
    echo "Error: Failed to read lake value from config file"
    exit 1
fi

# # Get gradient accumulation steps from config
# if ! GRAD_ACC="$(shyaml get-value gradient_accumulation_steps < "${CFG}")"; then
#     echo "Error: Failed to read gradient_accumulation_steps from config file"
#     exit 1
# fi

# # Calculate global batch size
# GLOBAL_BATCH=$((BATCH * GRAD_ACC * GPU))

# set deepspeed config file based on stage
DEEPSPEED_CONFIG_FILE="config/deepspeed/deepspeed_config_${DEEPSPEED_STAGE}.yaml"

# 构建运行参数
args="--config_file=${DEEPSPEED_CONFIG_FILE} train_unified_gen_und.py -c ${CFG} -z ${DEEPSPEED_STAGE} --pdb_debug"
echo $args

#CUDA_VISIBLE_DEVICES="0" 
accelerate launch --num_processes=${GPU} ${args}
