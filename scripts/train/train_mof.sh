#!/bin/bash
set -euo pipefail

echo "MOF training is not wired into the new HF-first component loader yet." >&2
echo "Register the custom vision tower/processor first, then call scripts/train/pretrain.sh or finetune.sh with overrides." >&2
exit 1
