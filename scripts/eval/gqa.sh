#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"
GQADIR="${GQADIR:-$EVAL_DIR/gqa}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.batch_generation \
        --adapter vqa \
        --model-path "$MODEL_PATH" \
        --question-file "$EVAL_DIR/gqa/$SPLIT.jsonl" \
        --image-folder "$EVAL_DIR/gqa/images" \
        --answers-file "$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait

output_file="$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl"
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto gqa \
    --prediction-file "$output_file" \
    --output-file "$GQADIR/testdev_balanced_predictions.json"

cd "$GQADIR"
python eval/eval.py --tier testdev_balanced
