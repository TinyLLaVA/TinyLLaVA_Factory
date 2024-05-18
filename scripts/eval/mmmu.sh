#!/bin/bash

MODEL_PATH="/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-final"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-final"
EVAL_DIR="/home/ai/data/llava/dataset/eval"

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/MMMU/eval

python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json
