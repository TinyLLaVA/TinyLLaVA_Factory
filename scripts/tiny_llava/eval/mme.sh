#!/bin/bash

MODEL_PATH="/root/autodl-tmp/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME
