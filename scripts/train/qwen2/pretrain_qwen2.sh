#!/bin/bash
set -euo pipefail

exec bash scripts/train/train_config.sh configs/train/models/qwen2_instruct_pretrain.yaml "$@"
