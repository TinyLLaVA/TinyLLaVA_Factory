#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/qwen2_base_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/qwen2_base_finetune.yaml "$@"
