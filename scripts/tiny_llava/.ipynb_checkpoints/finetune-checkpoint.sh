#!/bin/bash
if [ $# -ne 4 ]; then
    echo "Usage: $0 <LLM_VERSION> <VT_VERSION> <IMAGE_PATH> <DATA_PATH>"
    exit 1
fi

# Assign the arguments to variables
LLM_VERSION="$1"
VT_VERSION="$2"
DATA_PATH="$3"
IMAGE_PATH="$4"

# LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# VT_VERSION=openai/clip-vit-base-patch16
# DATA_PATH=/root/autodl-tmp/data/pretraining_data/LLaVA-Data/text_files/llava_v1_5_mix665k.json
# IMAGE_PATH=/root/autodl-tmp/data/pretraining_data/LLaVA-Data
VT_VARIANT="${VT_VERSION#*/}"


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower $VT_VERSION \
    --pretrain_mm_mlp_adapter ./checkpoints/tiny-llava-v1-1.1B-${VT_VARIANT}-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ./checkpoints/tiny-llava-v1-1.1B-${VT_VARIANT}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
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
    --run_name llava-finetune-tinyllama1.1B-${VT_VARIANT}
