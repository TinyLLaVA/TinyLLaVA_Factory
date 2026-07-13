#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.batch_generation \
        --adapter vqa \
        --model-path "$MODEL_PATH" \
        --question-file "$EVAL_DIR/vqav2/$SPLIT.jsonl" \
        --image-folder "$EVAL_DIR/vqav2/test2015" \
        --answers-file "$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait

output_file="$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/merge.jsonl"
mkdir -p "$(dirname "$output_file")"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto vqav2 \
    --prediction-file "$output_file" \
    --split-file "$EVAL_DIR/vqav2/llava_vqav2_mscoco_test2015.jsonl" \
    --output-file "$EVAL_DIR/vqav2/answers_upload/$SPLIT/$MODEL_NAME.json"
