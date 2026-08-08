# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}


IMAGE_DIR=/data/oss_bucket_1//tbstar_image_eval_results/Qwen3-VL-30B-A3B-Instruct_Qwen-Image-Edit-2509/wise/res_1024x1024/


python gpt_eval.py \
    --json_path data/cultural_common_sense.json \
    --output_dir ${IMAGE_DIR}/Results/cultural_common_sense \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --api_base "https://idealab.alibaba-inc.com/api/openai/v1"\
    --model "gpt-4o-0513-global" \
    --result_full ${IMAGE_DIR}/Results/cultural_common_sense_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl \
    --max_workers 20

python gpt_eval.py \
    --json_path data/natural_science.json \
    --output_dir ${IMAGE_DIR}/Results/natural_science \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --api_base "https://idealab.alibaba-inc.com/api/openai/v1"\
    --model "gpt-4o-0513-global" \
    --result_full ${IMAGE_DIR}/Results/natural_science_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/natural_science_scores_results.jsonl \
    --max_workers 20

python gpt_eval.py \
    --json_path data/spatio-temporal_reasoning.json \
    --output_dir ${IMAGE_DIR}/Results/spatio-temporal_reasoning \
    --image_dir ${IMAGE_DIR} \
    --api_key "" \
    --api_base "https://idealab.alibaba-inc.com/api/openai/v1"\
    --model "gpt-4o-0513-global" \
    --result_full ${IMAGE_DIR}/Results/spatio-temporal_reasoning_full_results.json \
    --result_scores ${IMAGE_DIR}/Results/spatio-temporal_reasoning_scores_results.jsonl \
    --max_workers 20


python Calculate.py \
    "${IMAGE_DIR}/Results/cultural_common_sense_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/natural_science_scores_results.jsonl" \
    "${IMAGE_DIR}/Results/spatio-temporal_reasoning_scores_results.jsonl" \
    --category all