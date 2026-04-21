# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}

export HF_ENDPOINT=https://hf-mirror.com


edited_images_dir=/data/oss_bucket_1//tbstar_image_eval_results/Qwen3-VL-30B-A3B-Instruct_Qwen-Image-Edit-2509/
model_name=gedit/system_prompt_version-v3
instruction_language=cn
save_dir=${edited_images_dir}/${model_name}/csv_results_${instruction_language}


task_types=(
    "background_change" "color_alter" "material_alter" "motion_change" "ps_human" 
    "style_change" "subject-add" "subject-remove" "subject-replace" "text_change" "tone_transfer"
)

# 循环启动后台任务
for task_type in "${task_types[@]}"; do
    echo "启动任务: ${task_type}"

    python run_gedit_score.py \
        --model_name "${model_name}" \
        --edited_images_dir "${edited_images_dir}" \
        --instruction_language "${instruction_language}" \
        --save_dir "${save_dir}" \
        --task_type "${task_type}" &
done

# 使用 wait 命令等待所有后台启动的任务执行完毕
echo "等待所有任务完成..."
wait
echo "所有任务已全部完成！"


python calculate_statistics.py \
    --model_name ${model_name} \
    --save_path ${save_dir} \
    --language ${instruction_language}

echo ${edited_images_dir}
echo ${model_name}
echo ${instruction_language}
echo ${save_dir}
echo "所有任务已全部完成！"