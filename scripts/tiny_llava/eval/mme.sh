#!/bin/bash

VERSION="$1"
LLM_VERSION="$2"
LLM_VARIANT="${LLM_VERSION#*/}"
VT_VERSION=google/siglip-so400m-patch14-384
VT_VARIANT="${VT_VERSION#*/}"

MODEL_PATH="./checkpoints/tiny-llava-${VERSION}-${LLM_VARIANT}-${VT_VARIANT}-finetune"

EVAL_DIR="/mnt/data/sata/ssd/dataset/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b
