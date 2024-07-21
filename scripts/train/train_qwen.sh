VERSION="$1"
PARAMETERS="$2"
INSTRUCT="$3"

DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset

LLM_VERSION=Qwen/Qwen${VERSION}-${PARAMETERS}B
if [ "$INSTRUCT" ] && ([ "$INSTRUCT" == "i" ] || [ "$INSTRUCT" == "instruct" ]); then
    LLM_VERSION=$LLM_VERSION-Instruct
fi
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu 
CONV_VERSION=phi #chat template for qwen is the same as that for phi
VERSION=base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048

BATCH_SIZE=16
ACC_STEPS=4
ZERO=2
bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$BATCH_SIZE" "$ACC_STEPS" "$ZERO"

BATCH_SIZE=4
ACC_STEPS=8

bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$BATCH_SIZE" "$ACC_STEPS" "$ZERO"
