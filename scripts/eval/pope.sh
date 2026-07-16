#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/pope.yaml}"
export MODEL_PATH MODEL_NAME EVAL_DIR

python -m tinyllava.eval.batch_generation \
    --config "$CONFIG_PATH"

python -m tinyllava.eval.tasks.auto.evaluation_auto pope \
    --annotation-dir "$EVAL_DIR/pope/coco" \
    --question-file "$EVAL_DIR/pope/llava_pope_test.jsonl" \
    --prediction-file "$EVAL_DIR/pope/answers/$MODEL_NAME.jsonl"
