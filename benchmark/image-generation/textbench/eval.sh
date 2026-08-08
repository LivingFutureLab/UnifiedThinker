# Get the directory of the currently executing script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${SCRIPT_DIR}

export HF_ENDPOINT=https://hf-mirror.com


#SAMPLE_FOLDER=/tmp//checkpoints/Qwen-Image-Edit-2509/textbench_zh
SAMPLE_FOLDER=/tmp//checkpoints/unified_sft_qwen_image_edit/qwen_image_edit_unified_alimama_zero2_bs1440_lr1e_05_20251010_01/ckpt/step-10000/textbench_zh

MODE=zh # en or zh
OUTPUT_DIR=${SAMPLE_FOLDER}


torchrun --nnodes=1 --node-rank=0 --nproc_per_node=8 \
    evaluate_text_reward.py \
    --sample_dir $SAMPLE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --mode $MODE

cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results.jsonl
rm $OUTPUT_DIR/results_chunk*.jsonl

python3 summary_scores.py $OUTPUT_DIR/results.jsonl --mode $MODE

echo ${SAMPLE_FOLDER}
echo ${MODE}
echo "评测结束!"
