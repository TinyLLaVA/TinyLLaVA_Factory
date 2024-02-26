#!/bin/bash

MODEL_PATH="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
