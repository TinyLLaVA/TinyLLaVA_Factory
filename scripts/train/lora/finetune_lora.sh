#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/lora_finetune.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

bash scripts/train/train_config.sh "$CONFIG_PATH" "$@"
