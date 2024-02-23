#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-base-patch16
VT_VARIANT="${VT_VERSION#*/}"


MODEL_PATH="./checkpoints/tiny-llava-v1-1.1B-${VT_VARIANT}-finetune"
MODEL_NAME="tiny-llava-v1-1.1B-${VT_VARIANT}"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode v1

python llava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
