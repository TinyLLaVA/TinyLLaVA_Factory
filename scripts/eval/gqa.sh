#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/gqa.yaml}"
GQADIR="${GQADIR:-$EVAL_DIR/gqa}"
export MODEL_PATH MODEL_NAME EVAL_DIR

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.batch_generation \
        --config "$CONFIG_PATH" \
        "output.answers_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" \
        "runtime.num_chunks=$CHUNKS" \
        "runtime.chunk_idx=$IDX" &
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

python -m tinyllava.eval.tasks.auto.evaluation_auto gqa \
    --prediction-file "$output_file" \
    --output-file "$GQADIR/testdev_balanced_predictions.json"

cd "$GQADIR"
python eval/eval.py --tier testdev_balanced
