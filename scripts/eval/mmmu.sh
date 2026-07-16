#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/mmmu.yaml}"
export MODEL_PATH MODEL_NAME EVAL_DIR

python -m tinyllava.eval.batch_generation \
    --config "$CONFIG_PATH"

python -m tinyllava.eval.tasks.auto.evaluation_auto mmmu \
    --prediction-file "$EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl" \
    --output-file "$EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json"

cd "$EVAL_DIR/MMMU/eval"

python main_eval_only.py --output_path "$EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json"
