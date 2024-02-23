#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-large-patch14-336
VT_VARIANT="${VT_VERSION#*/}"


MODEL_PATH="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
MODEL_NAME="tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}"
MODEL_BASE="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-base $MODEL_BASE \
    --conv-mode tiny_llama

python llava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
