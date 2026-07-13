#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter mmmu \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/MMMU/anns_for_eval.json" \
    --image-folder "$EVAL_DIR/MMMU/all_images" \
    --answers-file "$EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl" \
    --max-new-tokens 1024 \
    --temperature 0

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto mmmu \
    --prediction-file "$EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl" \
    --output-file "$EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json"

cd "$EVAL_DIR/MMMU/eval"

python main_eval_only.py --output_path "$EVAL_DIR/MMMU/answers/${MODEL_NAME}_output.json"
