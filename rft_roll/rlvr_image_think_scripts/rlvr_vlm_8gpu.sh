#!/bin/bash
set +x

cd "$(dirname "$(readlink -f "$0")")/.."

IP_ADDR=$(hostname -I | awk '{print $1}')

python rlvr_image_think_scripts/start_rlvr_vlm_pipeline.py  --config_name rlvr_vlm_8gpu 2>&1 | tee log_rlvr_vlm_8gpu_${IP_ADDR}.txt