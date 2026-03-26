#!/usr/bin/env bash
set -euo pipefail

echo "=== Surprisal Setup ==="
echo ""

# 1. Check Python
if ! python3 --version >/dev/null 2>&1; then
    echo "ERROR: Python 3.12+ required. Install from https://python.org"
    exit 1
fi
echo "Python: $(python3 --version)"

# 2. Check Docker
if ! docker --version >/dev/null 2>&1; then
    echo "ERROR: Docker required. Install from https://docs.docker.com/get-docker/"
    exit 1
fi
echo "Docker: $(docker --version)"

# 3. Check/install uv (Python package manager)
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 4. Clone repo if not already in it
if [ ! -f "pyproject.toml" ] || ! grep -q "surprisal" pyproject.toml 2>/dev/null; then
    echo ""
    echo "--- Cloning surprisal ---"
    git clone https://github.com/jbarnes850/surprisal.git
    cd surprisal
fi

# 5. Create venv and install
echo ""
echo "--- Installing surprisal ---"
uv sync
uv tool install --force --from . surprisal
# Reinstall CUDA torch if on GPU system (uv sync installs CPU-only)
if nvidia-smi &>/dev/null; then
    echo "GPU detected — installing CUDA PyTorch..."
    UV_HTTP_TIMEOUT=300 uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130 \
        --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
fi

# 6. Check/setup Claude CLI
echo ""
echo "--- Agent Setup ---"
if ! command -v claude &>/dev/null; then
    echo "Installing Claude CLI..."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || sudo npm install -g @anthropic-ai/claude-code
fi

if claude auth status 2>/dev/null | grep -q loggedIn; then
    echo "Claude CLI: authenticated"
else
    echo "Claude CLI: needs authentication"
    echo "Please run: claude auth login"
    echo "(Continuing setup — you can auth later)"
fi

if command -v codex &>/dev/null; then
    echo "Codex CLI: detected (optional)"
fi

# 7. Check GPU
echo ""
echo "--- GPU Detection ---"
GPU=false
HOST_OS="$(uname -s)"
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU detected: $GPU_NAME"
    GPU=true
else
    echo "No local NVIDIA GPU detected on $HOST_OS — CPU sandbox mode (use HF Jobs or remote GPU for scale-out)"
fi

# 8. Select sandbox image (built lazily on first run)
echo ""
echo "--- Sandbox Image Selection ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_IMAGE="surprisal-cpu:latest"
if [ "$GPU" = true ]; then
    SANDBOX_IMAGE="surprisal-gpu:latest"
fi
echo "Selected image: $SANDBOX_IMAGE"
if docker image inspect "$SANDBOX_IMAGE" >/dev/null 2>&1; then
    echo "Image already present locally."
else
    echo "Image not built yet — it will be built automatically on the first local run."
    echo "Optional prebuild:"
    if [ "$GPU" = true ]; then
        echo "  docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/"
    else
        echo "  docker build -t surprisal-cpu:latest -f sandbox/Dockerfile.cpu sandbox/"
    fi
fi

# 9. Credentials (optional)
echo ""
echo "--- Credentials (optional, press Enter to skip) ---"
read -rp "W&B API key: " WANDB_KEY
read -rp "HuggingFace token: " HF_TOKEN

# 10. Write config
LEGACY_DIR="$HOME/.surprisal"
XDG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/surprisal"
if [ -n "${SURPRISAL_HOME:-}" ]; then
    DATA_DIR="$SURPRISAL_HOME"
    CONFIG_DIR="$SURPRISAL_HOME"
elif [ -d "$LEGACY_DIR" ] && [ ! -d "$XDG_DIR" ]; then
    DATA_DIR="$LEGACY_DIR"
    CONFIG_DIR="$LEGACY_DIR"
else
    DATA_DIR="$XDG_DIR"
    CONFIG_DIR="$XDG_DIR"
fi
mkdir -p "$DATA_DIR"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.toml" << EOF
[sandbox]
backend = "auto"
image = "auto"
gpu = $GPU

[credentials]
wandb_api_key = "${WANDB_KEY:-}"
hf_token = "${HF_TOKEN:-}"
EOF
echo ""
echo "Config written to $CONFIG_DIR/config.toml"
echo "Exploration data will be stored under $DATA_DIR"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  uv run surprisal init --domain 'your research topic' --seed 'your hypothesis'"
echo "  uv run surprisal explore --budget 10"
