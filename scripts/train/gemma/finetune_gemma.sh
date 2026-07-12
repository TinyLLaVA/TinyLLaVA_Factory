#!/bin/bash
set -euo pipefail

exec bash scripts/train/train_config.sh configs/train/models/gemma_finetune.yaml "$@"
