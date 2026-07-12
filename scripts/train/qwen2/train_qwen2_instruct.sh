#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/qwen2_instruct_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/qwen2_instruct_finetune.yaml "$@"
