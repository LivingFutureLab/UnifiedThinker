# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}

result_dir=/data/oss_bucket_2//tbstar_image_eval_results/Qwen-Image-Edit-2509/risebench_think_v1/


#data="/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask.json"
# 只评测 temporal & causal
data="/data/oss_bucket_0//benchmarks/RISEBench/datav2_total_w_subtask_only_temp_cuasal.json"

python qwen_eval.py --data ${data} \
    --pretrained_model /tmp//checkpoints/Qwen2.5-VL-7B-Instruct/ \
    --input /data/oss_bucket_0//benchmarks/RISEBench/data/ \
    --output ${result_dir} \
    --prefix eval_risebench_by_ \
    --nproc 4