#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/vqav2.yaml}"
export MODEL_PATH MODEL_NAME EVAL_DIR

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.batch_generation \
        --config "$CONFIG_PATH" \
        "output.answers_file=$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" \
        "runtime.num_chunks=$CHUNKS" \
        "runtime.chunk_idx=$IDX" &
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

python -m tinyllava.eval.tasks.auto.evaluation_auto vqav2 \
    --prediction-file "$output_file" \
    --split-file "$EVAL_DIR/vqav2/llava_vqav2_mscoco_test2015.jsonl" \
    --output-file "$EVAL_DIR/vqav2/answers_upload/$SPLIT/$MODEL_NAME.json"
