#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train/default.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

python tinyllava/train/train.py --config "$CONFIG_PATH" "$@"
