DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset #finetune image dir

LLM_VERSION=stabilityai/stablelm-2-zephyr-1_6b # llm path in huggingface
VT_VERSION=openai/clip-vit-large-patch14-336 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template for stablelm is the same as that for phi
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
