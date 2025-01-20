cd /storage/qiguojunLab/fangxueji/Projects/iccv25

export WORLD_SIZE=8
export RANK=$1  # 从命令行参数接收 RANK
export OUTPUT_DIR="outputs_eval/geneval/momo_0.6b_t2i/checkpoint-20000/model"
echo "RANK is $RANK"

CUDA_VISIBLE_DEVICES=$RANK python geneval_sample.py \
    "/storage/qiguojunLab/fangxueji/DaddyVideo-final/geneval/geneval/prompts/evaluation_metadata.jsonl" \
    --model "outputs/momo_0.6b_t2i/checkpoint-20000/model" \
    --outdir $OUTPUT_DIR \
    --batch_size 1 \
    --steps 30 \
    --H 256 \
    --W 256 \
    --scale 5.5 \
    --rank $RANK \
    --world_size $WORLD_SIZE

# python geneval/evaluation/evaluate_images.py \
#     $OUTPUT_DIR \
#     --outfile "$OUTPUT_DIR/results.jsonl" \
#     --model-path "/storage/qiguojunLab/qiguojun/home/Models"

# python geneval/evaluation/summary_scores.py $OUTPUT_DIR/results.jsonl