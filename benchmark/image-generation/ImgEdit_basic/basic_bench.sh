
# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}

export HF_ENDPOINT=https://hf-mirror.com


#IMG_RESULT_PATH=/tmp//checkpoints/unified_sft_qwen_image_edit/qwen_image_edit_unified_alimama_zero2_bs1440_lr1e_05_20251010_01/ckpt/step-6000/imgedit/
#IMG_RESULT_PATH=/tmp//checkpoints/unified_sft_qwen_image_edit/qwen_image_edit_unified_alimama_zero2_bs1440_lr1e_05_20251010_01/ckpt/step-10000/imgedit/
#IMG_RESULT_PATH=/tmp//checkpoints/7b_freeze_20_full/ckpt/epoch-0-step-18500/imgedit/
IMG_RESULT_PATH=/tmp//checkpoints/Qwen-Image-Edit-2509/imgedit/

python basic_bench.py \
    --result_img_folder ${IMG_RESULT_PATH} \
    --edit_json ./basic_edit.json \
    --origin_img_root /data/oss_bucket_0//eval_open_source/edit/ImgEdit/singleturn/ \
    --num_processes 8 \
    --prompts_json ./prompts.json


python step1_get_avgscore.py \
    --result_json ${IMG_RESULT_PATH}/result.json \
    --average_score_json ${IMG_RESULT_PATH}/average_score.json


# 打开typescore.json就可以看到结果
python step2_typescore.py \
    --average_score_json ${IMG_RESULT_PATH}/average_score.json \
    --typescore_json ${IMG_RESULT_PATH}/typescore.json \
    --basic_edit ./basic_edit.json

echo "评测结束！"
echo ${IMG_RESULT_PATH}