#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter vqa \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/MME/llava_mme.jsonl" \
    --image-folder "$EVAL_DIR/MME/MME_Benchmark_release_version" \
    --answers-file "$EVAL_DIR/MME/answers/$MODEL_NAME.jsonl" \
    --temperature 0

cd "$EVAL_DIR/MME"

python convert_answer_to_mme.py --experiment "$MODEL_NAME"

cd eval_tool

python calculation.py --results_dir "answers/$MODEL_NAME"
