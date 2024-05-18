#!/bin/bash

MODEL_PATH="/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-phi-2-clip-vit-large-patch14-336-baseline-finetune/"
MODEL_NAME="tiny-llava-phi-2-clip-vit-large-patch14-336-baseline-finetune2"
EVAL_DIR="/home/ai/data/llava/dataset/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
