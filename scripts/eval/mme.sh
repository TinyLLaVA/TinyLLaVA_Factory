#!/bin/bash

MODEL_PATH="/home/jiajunlong/LLaVA/ying/checkpoints/tiny-llava-TinyLlama-1.1B-Chat-v1.0-clip-vit-large-patch14-336-tinyllama-llava-finetune"
MODEL_NAME="tiny-llava-TinyLlama-1.1B-Chat-v1.0-clip-vit-large-patch14-336-tinyllama-llava-finetune"
EVAL_DIR="/home/jiajunlong/llava_data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode llama

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME

