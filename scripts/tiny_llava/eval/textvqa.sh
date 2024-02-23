#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-large-patch14-336
VT_VARIANT="${VT_VERSION#*/}"

#MODEL_PATH="./checkpoints/tiny-tinyllava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
#MODEL_NAME="tiny-tinyllava-v1.5-${LLM_VARIANT}-${VT_VARIANT}"
#EVAL_DIR="/root/autodl-tmp/data/eval"
VERSION=type-3
MODEL_PATH="./checkpoints/tiny-llava-${VERSION}-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
MODEL_BASE="./checkpoints/tiny-llava-${VERSION}-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
#MODEL_BASE=$LLM_VERSION
MODEL_NAME="tiny-llava-${VERSION}-${LLM_VARIANT}-${VT_VARIANT}"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-base $MODEL_BASE \
    --conv-mode v1

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl
