#!/bin/bash
set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<<"$gpu_list"

CHUNKS=${#GPULIST[@]}

ROOT=$(
  cd "$(dirname $0)/../.." || exit
  pwd
)
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${ROOT}/dataset/gqa"

# MODEL_PATH="/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune/"
# MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
MODEL_PATH="${HOME}/.cache/huggingface/hub/models--Zhang199--TinyLLaVA-Qwen2-0.5B-SigLIP/snapshots/6aef66ed2e0125f57a5ec562fe3c0bf1204d8fa3"
MODEL_NAME="Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP"
EVAL_DIR="${ROOT}/dataset"

for IDX in $(seq 0 $((CHUNKS - 1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
    --image-folder $EVAL_DIR/gqa/images \
    --answers-file $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode qwen2_base # &
done

wait

output_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
>"$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
  cat $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >>"$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
ls
python eval.py --tier testdev_balanced --questions testdev_balanced_questions.json --consistency
