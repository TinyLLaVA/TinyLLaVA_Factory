#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/phi_share_base_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/phi_share_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/phi_share_finetune.yaml "$@"
