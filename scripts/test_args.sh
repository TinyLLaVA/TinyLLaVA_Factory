# original way of passing arguments to a script

DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm

bash scripts/dummy_launch_pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"


# new way of passing arguments
deepspeed --include localhost:4,5,6,7 --master_port 29501 scripts/test_new_args.py \
    --config_path ./configs/training/phi-pretrain.yml \
    --deepspeed ./configs/zero3.json