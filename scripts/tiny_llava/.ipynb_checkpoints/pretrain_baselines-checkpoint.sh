LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSIONS=(openai/clip-vit-large-patch14-336 openai/clip-vit-large-patch14 openai/clip-vit-base-patch16 openai/clip-vit-large-patch32)
DATA_PATH=/root/autodl-tmp/data/llava/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/root/autodl-tmp/data/llava/llava_pretrain/images


for VT_VERSION in "${VT_VERSIONS[@]}"; do
    bash scripts/tiny_llava/pretrain.sh "$LLM_VERSION" "$VT_VERSION" "$DATA_PATH" "$IMAGE_PATH"
done