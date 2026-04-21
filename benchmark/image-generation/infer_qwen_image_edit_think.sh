#!/bin/bash
set -x  # 启用命令回显方便调试

# 选择 GPU 数量
GPUS=8   # 改成你本地的 GPU 数量，比如 1 表示单卡，8 表示多卡

# 入口文件
ENTRY_FILE="benchmark/image-generation/infer_qwen_image_edit_think.py"

# 数据集 & 模式
dataset="gedit"
think=0

# 模型路径（改成本地路径）
pretrained_model="/tmp/checkpoints/Qwen-Image-Edit-2509"

# 训练 checkpoint（改成本地路径）
train_ckpt_file="/tmp/checkpoints/qwen_image_edit_think_experiments/qwen_image_edit_think_zero2_bs1024_lr1e_05_20251024_124721/ckpt/step-3000/"

# LoRA 配置（保持原有参数）
text_encoder_lora=1
transformer_lora=1

# 环境变量（可选）
export NCCL_IB_QPS_PER_CONNECTION=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export NCCL_DEBUG=INFO

# 参数数组
options_array=(
    --pretrained_model="${pretrained_model}"
    --train_ckpt_file="${train_ckpt_file}"
    --text_encoder_lora=${text_encoder_lora}
    --transformer_lora=${transformer_lora}
    --dataset=${dataset}
    --think=${think}
)

# ---------------- 多GPU运行 ----------------
if [ ${GPUS} -gt 1 ]; then
    torchrun \
        --nproc_per_node=${GPUS} \
        --master_port=15432 \
        ${ENTRY_FILE} \
        ${options_array[@]}
else
    # ---------------- 单GPU运行 ----------------
    python ${ENTRY_FILE} ${options_array[@]}
fi
