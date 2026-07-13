#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter pope \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/pope/llava_pope_test.jsonl" \
    --image-folder "$EVAL_DIR/pope/val2014" \
    --answers-file "$EVAL_DIR/pope/answers/$MODEL_NAME.jsonl" \
    --temperature 0

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto pope \
    --annotation-dir "$EVAL_DIR/pope/coco" \
    --question-file "$EVAL_DIR/pope/llava_pope_test.jsonl" \
    --prediction-file "$EVAL_DIR/pope/answers/$MODEL_NAME.jsonl"
