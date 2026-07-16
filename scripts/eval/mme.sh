#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"
CONFIG_PATH="${EVAL_CONFIG:-$REPO_ROOT/configs/eval/mme.yaml}"
export MODEL_PATH MODEL_NAME EVAL_DIR

python -m tinyllava.eval.batch_generation \
    --config "$CONFIG_PATH"

cd "$EVAL_DIR/MME"

python convert_answer_to_mme.py --experiment "$MODEL_NAME"

cd eval_tool

python calculation.py --results_dir "answers/$MODEL_NAME"
