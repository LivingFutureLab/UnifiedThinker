#!/bin/bash
set +x

export RAY_DEBUG=legacy
export RAY_DEBUG_POST_MORTEM=1

python src_rl/rlvr_image_think/start_rlvr_image_think_pipeline.py  --config_name rlvr_image_think_debug
