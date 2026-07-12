#!/bin/bash
set -euo pipefail

bash scripts/train/train_config.sh configs/train/models/phi_lora_pretrain.yaml "$@"
bash scripts/train/train_config.sh configs/train/models/phi_lora_finetune.yaml "$@"
