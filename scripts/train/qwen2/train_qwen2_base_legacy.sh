#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-all}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs used for training.}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "$GPU_COUNT" -eq 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES does not contain any GPU indices." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MASTER_PORT="${MASTER_PORT:-29501}"
PRETRAIN_CONFIG="configs/train/models/qwen2_base_legacy_pretrain.yaml"
FINETUNE_CONFIG="configs/train/models/qwen2_base_legacy_finetune.yaml"
PRETRAIN_MICRO_BATCH=16
FINETUNE_MICRO_BATCH=4
PRETRAIN_GLOBAL_BATCH=256
FINETUNE_GLOBAL_BATCH=128

export TINYLLAVA_PRETRAIN_DATA="${TINYLLAVA_PRETRAIN_DATA:-/data/vlm/llava_data/text_files/blip_laion_cc_sbu_558k.json}"
export TINYLLAVA_PRETRAIN_IMAGE_FOLDER="${TINYLLAVA_PRETRAIN_IMAGE_FOLDER:-/data/vlm/llava_data/llava/llava_pretrain/images}"
export TINYLLAVA_FINETUNE_DATA="${TINYLLAVA_FINETUNE_DATA:-/data/vlm/llava_data/text_files/llava_v1_5_mix665k.json}"
export TINYLLAVA_FINETUNE_IMAGE_FOLDER="${TINYLLAVA_FINETUNE_IMAGE_FOLDER:-/data/vlm/llava_data}"
export TINYLLAVA_PRETRAIN_OUTPUT="${TINYLLAVA_PRETRAIN_OUTPUT:-output/tinyllava-qwen2-base-legacy-pretrain}"
export TINYLLAVA_PRETRAINED_MODEL="${TINYLLAVA_PRETRAINED_MODEL:-$TINYLLAVA_PRETRAIN_OUTPUT}"
export TINYLLAVA_FINETUNE_OUTPUT="${TINYLLAVA_FINETUNE_OUTPUT:-output/tinyllava-qwen2-base-legacy-finetune}"

gradient_accumulation() {
    local global_batch="$1"
    local micro_batch="$2"
    local denominator=$((GPU_COUNT * micro_batch))
    if (( global_batch % denominator != 0 )); then
        echo "Global batch $global_batch is not divisible by $GPU_COUNT GPUs x micro batch $micro_batch." >&2
        return 2
    fi
    echo $((global_batch / denominator))
}

PRETRAIN_ACCUMULATION="$(gradient_accumulation "$PRETRAIN_GLOBAL_BATCH" "$PRETRAIN_MICRO_BATCH")"
FINETUNE_ACCUMULATION="$(gradient_accumulation "$FINETUNE_GLOBAL_BATCH" "$FINETUNE_MICRO_BATCH")"

preflight() {
    local required_paths=(
        "$PYTHON_BIN"
        "$TINYLLAVA_PRETRAIN_DATA"
        "$TINYLLAVA_PRETRAIN_IMAGE_FOLDER"
        "$TINYLLAVA_FINETUNE_DATA"
        "$TINYLLAVA_FINETUNE_IMAGE_FOLDER"
    )
    for required_path in "${required_paths[@]}"; do
        if [[ ! -e "$required_path" ]]; then
            echo "Required path does not exist: $required_path" >&2
            return 2
        fi
    done
    if ! "$PYTHON_BIN" -c 'import deepspeed' >/dev/null 2>&1; then
        echo "Missing required dependency deepspeed in $PYTHON_BIN." >&2
        return 2
    fi
    echo "pretrain: GPUs=$GPU_COUNT, micro_batch=$PRETRAIN_MICRO_BATCH, accumulation=$PRETRAIN_ACCUMULATION, global_batch=$PRETRAIN_GLOBAL_BATCH"
    echo "finetune: GPUs=$GPU_COUNT, micro_batch=$FINETUNE_MICRO_BATCH, accumulation=$FINETUNE_ACCUMULATION, global_batch=$FINETUNE_GLOBAL_BATCH"
}

run_stage() {
    local config_path="$1"
    local accumulation="$2"
    local command=(
        "$PYTHON_BIN"
        -m torch.distributed.run
        "--nproc_per_node=$GPU_COUNT"
        "--master_port=$MASTER_PORT"
        tinyllava/train/train.py
        --config "$config_path"
        "training.gradient_accumulation_steps=$accumulation"
    )
    "${command[@]}"
}

case "$STAGE" in
    check)
        preflight
        ;;
    pretrain)
        preflight
        run_stage "$PRETRAIN_CONFIG" "$PRETRAIN_ACCUMULATION"
        ;;
    finetune)
        preflight
        if [[ ! -f "$TINYLLAVA_PRETRAINED_MODEL/config.json" ]]; then
            echo "Pretraining output is incomplete: $TINYLLAVA_PRETRAINED_MODEL" >&2
            exit 2
        fi
        run_stage "$FINETUNE_CONFIG" "$FINETUNE_ACCUMULATION"
        ;;
    all)
        preflight
        run_stage "$PRETRAIN_CONFIG" "$PRETRAIN_ACCUMULATION"
        run_stage "$FINETUNE_CONFIG" "$FINETUNE_ACCUMULATION"
        ;;
    *)
        echo "Usage: $0 {check|pretrain|finetune|all}" >&2
        exit 2
        ;;
esac
