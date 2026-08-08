# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}


image_path=/data/oss_bucket_1//tbstar_image_eval_results/Qwen-Image-Edit-2509/prism_bench/en/

api_key=''
base_url="https://idealab.alibaba-inc.com/api/openai/v1"
api_model="gpt-41-0414-global"



echo "开始并行执行评估任务..."
for category in "imagination" "entity" "text_rendering" "style" "affection" "composition" "long_text"
do
    echo "启动对 [${category}] 的评估..."
    python eval_gpt41.py \
        --image_path ${image_path} \
        --api_key ${api_key} \
        --base_url ${base_url} \
        --api_model ${api_model} \
        --category ${category} &
done
wait
echo "所有评估任务已全部完成！"


echo "\n\n最后一起执行一次，打印全量的结果\n"
python eval_gpt41.py \
    --image_path ${image_path} \
    --api_key ${api_key} \
    --base_url ${base_url} \
    --api_model ${api_model}
