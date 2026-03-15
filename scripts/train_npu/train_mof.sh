
ROOT=/root/TinyLLaVA_Factory
DATASET_ROOT=${ROOT}/dataset
DATA_PATH=${DATASET_ROOT}/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=${DATASET_ROOT}/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=${DATASET_ROOT}/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=${DATASET_ROOT}

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
VT_VERSION=mof:openai/clip-vit-large-patch14
VT_VERSION2=mof:facebook/dinov2-large
CN_VERSION=mof_mlp
CONV_VERSION=llama
VERSION=llama-mof-base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048


bash scripts/train_npu/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train_npu/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
