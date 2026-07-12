#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/tinyllama_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/tinyllama_finetune.yaml "$@"
