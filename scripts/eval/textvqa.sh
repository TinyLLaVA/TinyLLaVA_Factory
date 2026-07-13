#!/bin/bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a TinyLLaVA checkpoint path or HF repo id.}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DIR="${EVAL_DIR:-./eval}"

python -m tinyllava.eval.batch_generation \
    --adapter vqa \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "$EVAL_DIR/textvqa/train_images" \
    --answers-file "$EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl" \
    --temperature 0

python -m tinyllava.eval.dataset_adapters.auto.evaluation_auto textvqa \
    --annotation-file "$EVAL_DIR/textvqa/TextVQA_0.5.1_val.json" \
    --prediction-file "$EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl"
