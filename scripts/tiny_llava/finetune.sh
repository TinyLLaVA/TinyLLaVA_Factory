LLM_VERSION=microsoft/phi-2
VT_VERSION=google/siglip-so400m-patch14-384

DATA_PATH=/path/to/your/llava_v1_5_mix665k.json
IMAGE_PATH=/path/to/your/data
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --pretrain_mm_mlp_adapter ./checkpoints/tiny-llava-base-${LLM_VARIANT}-${VT_VARIANT}-pretrain/mm_projector.bin \
    --output_dir ./checkpoints/tiny-llava-base-${LLM_VARIANT}-${VT_VARIANT}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name tiny-llava-base-finetune-${LLM_VARIANT}-${VT_VARIANT}
