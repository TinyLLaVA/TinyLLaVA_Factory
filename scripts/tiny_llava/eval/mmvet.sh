#!/bin/bash

MODEL_PATH="./checkpoints/tiny-llava-v1-1.1B-clip-vit-large-patch14-336-finetune"
MODEL_NAME="tiny-llava-v1-1.1b"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/tinyllava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $MODEL_PATH/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json

