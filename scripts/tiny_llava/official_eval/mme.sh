#!/bin/bash

MODEL_PATH="/root/autodl-tmp/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B"
EVAL_DIR="/root/autodl-tmp/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file .$EVAL_DIR/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment TinyLLaVA-3.1B

cd eval_tool

python calculation.py --results_dir answers/TinyLLaVA-3.1B
