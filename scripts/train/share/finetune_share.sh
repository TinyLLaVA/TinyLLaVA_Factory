#!/bin/bash
set -euo pipefail

exec bash scripts/train/train_config.sh configs/train/models/phi_share_finetune.yaml "$@"
