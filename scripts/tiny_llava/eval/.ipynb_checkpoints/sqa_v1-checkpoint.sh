#!/bin/bash

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-base-patch16
VT_VARIANT="${VT_VERSION#*/}"


MODEL_PATH="./checkpoints/tiny-llava-v1-1.1B-${VT_VARIANT}-finetune"
MODEL_NAME="tiny-llava-v1-1.1B-${VT_VARIANT}"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v1

python llava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json

