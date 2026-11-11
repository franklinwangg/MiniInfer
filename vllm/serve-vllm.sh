#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$HOME/.venvs/miniinfer-vllm"
VLLM="$VENV_DIR/bin/vllm"

if [ ! -x "$VLLM" ]; then
    echo "vLLM is not installed. Run: bash setup-vllm.sh" >&2
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
exec "$VLLM" serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype half \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager
