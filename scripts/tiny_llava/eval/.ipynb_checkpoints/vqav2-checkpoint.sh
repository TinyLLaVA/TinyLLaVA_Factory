#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

#LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
#LLM_VARIANT="${LLM_VERSION#*/}"
#
#VT_VERSION=openai/clip-vit-large-patch14-336
#VT_VARIANT="${VT_VERSION#*/}"
#
#MODEL_PATH="./checkpoints/tiny-llava-v1-${LLM_VARIANT}-${VT_VARIANT}-finetune"
#MODEL_NAME="tiny-llava-v1-1.1b-sharegpt4v"
#EVAL_DIR="/root/autodl-tmp/data/eval"
LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VARIANT="${LLM_VERSION#*/}"

VT_VERSION=openai/clip-vit-large-patch14-336
VT_VARIANT="${VT_VERSION#*/}"

MODEL_PATH="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-finetune-lora"
MODEL_BASE="./checkpoints/tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}-pretrain"
MODEL_NAME="tiny-llava-v1.5-${LLM_VARIANT}-${VT_VARIANT}"
EVAL_DIR="/root/autodl-tmp/data/eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/vqav2/test2015 \
        --answers-file $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --model-base $MODEL_BASE
        --conv-mode tiny_llama &
done

wait

output_file=$EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/vqav2/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $MODEL_NAME --dir $EVAL_DIR/vqav2

