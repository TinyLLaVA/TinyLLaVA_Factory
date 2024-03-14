LLM_VERSION=microsoft/phi-2
VT_VERSION=google/siglip-so400m-patch14-384

DATA_PATH=/path/to/your/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/path/to/your/images
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --tune_entire_model False \
    --fp16 True \
    --output_dir ./checkpoints/tiny-llava-base-"${LLM_VARIANT}"-"${VT_VARIANT}"-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --run_name tiny-llava-base-pretrain-"${LLM_VARIANT}"-"${VT_VARIANT}"