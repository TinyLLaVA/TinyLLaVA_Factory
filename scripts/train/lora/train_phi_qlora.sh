#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/phi_qlora_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/phi_qlora_finetune.yaml "$@"
