#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version; then
    echo "Run this script inside Ubuntu on WSL2." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "The NVIDIA GPU is not visible in WSL. Update the Windows NVIDIA driver and restart WSL." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required. Install it with: sudo apt-get update && sudo apt-get install -y curl" >&2
    exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
    echo "A C compiler is required by vLLM's CUDA kernels. Install it with: sudo apt-get update && sudo apt-get install -y build-essential" >&2
    exit 1
fi

if ! nvidia-smi; then
    echo "nvidia-smi failed inside WSL; CUDA GPU access is not ready." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

VENV_DIR="$HOME/.venvs/miniinfer-vllm"
mkdir -p "$(dirname "$VENV_DIR")"
uv venv --python 3.12 --seed --managed-python "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" vllm --torch-backend=auto

"$VENV_DIR/bin/python" -c \
    'import torch, vllm; assert torch.cuda.is_available(), "CUDA is not available in WSL"; print(f"vLLM {vllm.__version__}; GPU: {torch.cuda.get_device_name(0)}")'
