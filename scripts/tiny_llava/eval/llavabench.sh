#!/bin/bash

MODEL_PATH="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B"
EVAL_DIR="./playground/data/eval"


python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $EVAL_DIR/llava-bench-in-the-wild/images \
    --answers-file $EVAL_DIR/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode v1

mkdir -p $EVAL_DIR/llava-bench-in-the-wild/reviews