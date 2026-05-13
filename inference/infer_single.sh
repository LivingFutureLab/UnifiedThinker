#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

DEFAULT_PRETRAINED="model/Qwen-Image-Edit-2509"
DEFAULT_PROCESSOR="model/UnifiedThinker-7B"

# 用法: ./run_editor.sh [MODEL_PATH] [PROCESSOR_PATH] [CKPT_PATH]
PRETRAINED_MODEL=${1:-$DEFAULT_PRETRAINED}
VLM_PROCESSOR=${2:-$DEFAULT_PROCESSOR}

echo "------------------------------------------------"
echo "🚀 Starting Image Editor..."
echo "📍 Pretrained Model: $PRETRAINED_MODEL"
echo "📍 VLM Processor:    $VLM_PROCESSOR"
if [ ! -z "$TRAIN_CKPT" ]; then
    echo "📍 Checkpoint:       $TRAIN_CKPT"
fi
echo "------------------------------------------------"
export PYTHONPATH=$(pwd):${PYTHONPATH}

python3  inference/infer_single.py \
    --model_path "$PRETRAINED_MODEL" \
    --processor_path "$VLM_PROCESSOR" \
    --ckpt_path "$TRAIN_CKPT"