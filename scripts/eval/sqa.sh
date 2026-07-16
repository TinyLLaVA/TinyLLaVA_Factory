#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/scienceqa.yaml}"
export MODEL_PATH MODEL_NAME EVAL_DIR

python -m tinyllava.eval.batch_generation \
    --config "$CONFIG_PATH"

python -m tinyllava.eval.tasks.auto.evaluation_auto scienceqa \
    --base-dir "$EVAL_DIR/scienceqa" \
    --prediction-file "$EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl" \
    --output-file "$EVAL_DIR/scienceqa/answers/${MODEL_NAME}_output.jsonl" \
    --output-result-file "$EVAL_DIR/scienceqa/answers/${MODEL_NAME}_result.json"
