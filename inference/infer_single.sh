#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Override the interpreter with PYTHON=<python>; defaults to python3
PYTHON=${PYTHON:-python3}

DEFAULT_PRETRAINED="model/Qwen-Image-Edit-2509"
DEFAULT_PROCESSOR="model/UnifiedThinker-7B"

# Usage: ./infer_single.sh [MODEL_PATH] [PROCESSOR_PATH] [THINKER_PATH]
#   TRAIN_CKPT=<dir> is optional, to load a checkpoint from thinker-editor training
PRETRAINED_MODEL=${1:-$DEFAULT_PRETRAINED}
VLM_PROCESSOR=${2:-$DEFAULT_PROCESSOR}
# Thinker weights; defaults to the same dir as the processor (i.e. UnifiedThinker-7B)
THINKER_PATH=${3:-$VLM_PROCESSOR}

echo "------------------------------------------------"
echo "🚀 Starting Image Editor..."
echo "📍 Pretrained Model: $PRETRAINED_MODEL"
echo "📍 VLM Processor:    $VLM_PROCESSOR"
echo "📍 Thinker:          $THINKER_PATH"
if [ ! -z "$TRAIN_CKPT" ]; then
    echo "📍 Checkpoint:       $TRAIN_CKPT"
fi
echo "------------------------------------------------"
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Only pass --ckpt_path when TRAIN_CKPT is non-empty, to avoid an empty string
CKPT_ARG=""
if [ ! -z "$TRAIN_CKPT" ]; then
    CKPT_ARG="--ckpt_path $TRAIN_CKPT"
fi

# Optional: set IMAGE and PROMPT for non-interactive single-shot mode; if neither is set it enters the interactive loop.
#   IMAGE=<image path> PROMPT=<edit instruction> [OUTPUT=<output path>] [NO_THINKING=1] ./infer_single.sh
SINGLE_ARGS=""
[ -n "$IMAGE" ]       && SINGLE_ARGS="$SINGLE_ARGS --image $IMAGE"
[ -n "$PROMPT" ]      && SINGLE_ARGS="$SINGLE_ARGS --prompt \"$PROMPT\""
[ -n "$OUTPUT" ]      && SINGLE_ARGS="$SINGLE_ARGS --output $OUTPUT"
[ -n "$NO_THINKING" ] && SINGLE_ARGS="$SINGLE_ARGS --no_thinking"

# Use eval to correctly handle spaces/quotes inside PROMPT
eval "$PYTHON" inference/infer_single.py \
    --model_path "\"$PRETRAINED_MODEL\"" \
    --processor_path "\"$VLM_PROCESSOR\"" \
    --thinker_path "\"$THINKER_PATH\"" \
    $CKPT_ARG $SINGLE_ARGS