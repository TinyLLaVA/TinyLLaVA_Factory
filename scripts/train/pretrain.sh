#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/pretrain.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

bash scripts/train/train_config.sh "$CONFIG_PATH" "$@"
