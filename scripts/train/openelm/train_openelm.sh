#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/openelm_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/openelm_finetune.yaml "$@"
