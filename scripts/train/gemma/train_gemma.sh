#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/gemma_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/gemma_finetune.yaml "$@"
