#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter vqa \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/mm-vet/llava-mm-vet.jsonl" \
    --image-folder "$EVAL_DIR/mm-vet/images" \
    --answers-file "$EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl" \
    --temperature 0

mkdir -p "$EVAL_DIR/mm-vet/results"

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto mmvet \
    --prediction-file "$EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl" \
    --output-file "$EVAL_DIR/mm-vet/results/$MODEL_NAME.json"
