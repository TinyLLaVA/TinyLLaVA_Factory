#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-base-patch16
VT_VARIANT="${VT_VERSION#*/}"

MODEL_PATH="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
MODEL_NAME="tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}"
MODEL_BASE="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-base $MODEL_BASE \
    --conv-mode tiny_llama

python -m llava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl
