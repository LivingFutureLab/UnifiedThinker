#!/bin/bash


export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

PRETRAINED_MODEL="model/Qwen-Image-Edit-2509"
VLM_PROCESSOR="model/UnifiedThinker-7B"

echo "------------------------------------------------"
echo "🎨 Starting UnifiedThinker Gradio Demo..."
echo "📍 Project Root: $PROJECT_ROOT"
echo "📍 Pretrained:   $PRETRAINED_MODEL"
echo "📍 VLM:          $VLM_PROCESSOR"
echo "🌐 Gradio will be available on: http://0.0.0.0:7860"
echo "------------------------------------------------"


python3 inference/infer_gradio.py \
    --model_path "$PRETRAINED_MODEL" \
    --processor_path "$VLM_PROCESSOR"