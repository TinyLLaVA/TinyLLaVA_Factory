LLM_VERSION=susnato/phi-1_5_dev
# LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSIONS=(openai/clip-vit-large-patch14-336)
#DATA_PATH=/root/autodl-tmp/data/text_files/really_cleaned_share-captioner_coco_lcs_sam_1246k_1107.json
DATA_PATH=/root/autodl-tmp/data/llava/blip_laion_cc_sbu_558k.json
#FINETUNE_DATA_PATH=/root/autodl-tmp/data/text_files/cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
FINETUNE_DATA_PATH=/root/autodl-tmp/data/text_files/llava_v1_5_mix665k.json
#IMAGE_PATH=/root/autodl-tmp/data/
#FINETUNE_IMAGE_PATH=/root/autodl-tmp/data
IMAGE_PATH=/root/autodl-tmp/data/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/root/autodl-tmp/data


for VT_VERSION in "${VT_VERSIONS[@]}"; do
     bash scripts/tiny_llava/pretrain.sh "$LLM_VERSION" "$VT_VERSION" "$DATA_PATH" "$IMAGE_PATH"
    bash scripts/tiny_llava/finetune_lora.sh "$LLM_VERSION" "$VT_VERSION" "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH"
done

