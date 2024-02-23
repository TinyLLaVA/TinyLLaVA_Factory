LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSIONS=(openai/clip-vit-large-patch14 openai/clip-vit-base-patch16)
DATA_PATH=/root/autodl-tmp/data/llava/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/root/autodl-tmp/data/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/root/autodl-tmp/data/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/root/autodl-tmp/data

# bash scripts/tiny_llava/finetune.sh "$LLM_VERSION" openai/clip-vit-large-patch14-336 "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH"
for VT_VERSION in "${VT_VERSIONS[@]}"; do
    # bash scripts/tiny_llava/pretrain.sh "$LLM_VERSION" "$VT_VERSION" "$DATA_PATH" "$IMAGE_PATH"
    bash scripts/tiny_llava/finetune.sh "$LLM_VERSION" "$VT_VERSION" "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH"
done

bash scripts/tiny_llava/pretrain.sh "$LLM_VERSION" openai/clip-vit-base-patch32 "$DATA_PATH" "$IMAGE_PATH"
bash scripts/tiny_llava/finetune.sh "$LLM_VERSION" openai/clip-vit-base-patch32 "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH"