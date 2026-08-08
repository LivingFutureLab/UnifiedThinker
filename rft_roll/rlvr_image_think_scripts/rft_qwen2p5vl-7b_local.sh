#!/bin/bash
set +x

cd "$(dirname "$(readlink -f "$0")")/.."


IP_ADDR=$(hostname -I | awk '{print $1}')

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

python rlvr_image_think_scripts/start_rlvr_image_think_pipeline.py  \
    --config_name rft_qwen2p5vl-7b_local --run_timestamp ${TIMESTAMP} 2>&1 | tee log_rft_qwen2p5vl-7b_local_${IP_ADDR}.txt
