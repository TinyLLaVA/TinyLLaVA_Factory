#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

BENCHMARK="${1:?Usage: $0 BENCHMARK}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/output/tinyllava-qwen2-base-legacy-finetune}"
export MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
export EVAL_DIR="${EVAL_DIR:-/data/vlm/llava_data/eval}"
export EVAL_CHAT_TEMPLATE_PATH="${EVAL_CHAT_TEMPLATE_PATH:-$REPO_ROOT/configs/chat_templates/qwen2_base_legacy.jinja}"

case "$BENCHMARK" in
    vqav2|gqa|textvqa|pope|mme|mmvet|mmmu)
        unset EVAL_CONFIG
        exec bash "$REPO_ROOT/scripts/eval/$BENCHMARK.sh"
        ;;
    scienceqa|sqa)
        export EVAL_CONFIG="$REPO_ROOT/configs/eval/scienceqa_qwen2_base_legacy.yaml"
        exec bash "$REPO_ROOT/scripts/eval/sqa.sh"
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK" >&2
        exit 2
        ;;
esac
