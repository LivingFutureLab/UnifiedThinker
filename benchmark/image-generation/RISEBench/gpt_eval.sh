# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}

#result_dir=/tmp//private_domain//checkpoints/tbstar_image_think/ablation_und_task/qwenvl_und_zero2_bs512_lr1e_05_20251022_20/ckpt/step-800/risebench_think/
#result_dir=/data/oss_bucket_1//tbstar_image_eval_results/Qwen-Image-Edit-2509/risebench_think_v1/
result_dir="/data/oss_bucket_1//tbstar_image_eval_results/Qwen3-VL-30B-A3B-Instruct_Qwen-Image-Edit-2509/risebench/thinker_prompt_version-v3-reflector_prompt_version-v1/"


data="/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask.json"
# 只评测 temporal & causal
#data="/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask_only_temp_cuasal.json"
# 评测 wo logical_reasoning
#data="/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask_wo_logical.json"

python gpt_eval.py --data ${data} \
    --input /data/oss_bucket_0//benchmarks/RISEBench/data/ \
    --output ${result_dir} \
    --nproc 10
