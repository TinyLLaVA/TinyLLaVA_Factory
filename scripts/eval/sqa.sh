#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter scienceqa \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/scienceqa/llava_test_CQM-A.json" \
    --image-folder "$EVAL_DIR/scienceqa/images/test" \
    --answers-file "$EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl" \
    --max-new-tokens 1024 \
    --single-pred-prompt \
    --temperature 0

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto scienceqa \
    --base-dir "$EVAL_DIR/scienceqa" \
    --prediction-file "$EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl" \
    --output-file "$EVAL_DIR/scienceqa/answers/${MODEL_NAME}_output.jsonl" \
    --output-result-file "$EVAL_DIR/scienceqa/answers/${MODEL_NAME}_result.json"
