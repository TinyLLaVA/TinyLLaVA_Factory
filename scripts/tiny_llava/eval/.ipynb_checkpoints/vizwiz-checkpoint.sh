#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-large-patch14-336
VT_VARIANT="${VT_VERSION#*/}"

# MODEL_PATH="./checkpoints/tiny-llava-v1-${LLM_VARIANT}-${VT_VARIANT}-finetune"
# MODEL_NAME="tiny-llava-v1-1.1b-sharegpt4v"
# MODEL_BASE="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
MODEL_PATH="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
MODEL_BASE="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
MODEL_NAME="tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --image-folder $EVAL_DIR/vizwiz/test \
    --answers-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-base $MODEL_BASE \
    --conv-mode tiny_llama

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --result-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --result-upload-file $EVAL_DIR/vizwiz/answers_upload/$MODEL_NAME.json
