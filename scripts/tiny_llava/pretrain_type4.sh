#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <LLM_VERSION> <VT_VERSION> <DATA_PATH> <IMAGE_PATH> <VERSION>"
    exit 1
fi

# Assign the arguments to variables
LLM_VERSION="$1"
VT_VERSION="$2"
DATA_PATH="$3"
IMAGE_PATH="$4"
VERSION="$5"

# LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# VT_VERSION=openai/clip-vit-base-patch16
# DATA_PATH=/root/autodl-tmp/data/tinyllava/blip_laion_cc_sbu_558k.json
# IMAGE_PATH=/root/autodl-tmp/data/tinyllava/llava_pretrain/images
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir ./checkpoints/tiny-llava-"${VERSION}"-"${LLM_VARIANT}"-"${VT_VARIANT}"-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 30 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name tiny-llava-"${VERSION}"-pretrain-"${LLM_VARIANT}"-"${VT_VARIANT}"
