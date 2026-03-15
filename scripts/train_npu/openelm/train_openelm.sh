DATASET_ROOT=/root/TinyLLaVA_Factory/dataset

DATA_PATH=${DATASET_ROOT}/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=${DATASET_ROOT}/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=${DATASET_ROOT}/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=${DATASET_ROOT}

LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048


bash scripts/train_npu/openelm/pretrain_openelm.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
#bash scripts/train_npu/openelm/finetune_openelm.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
