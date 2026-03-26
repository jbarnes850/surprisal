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
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU detected: $GPU_NAME"
    GPU=true
else
    echo "No GPU detected — stats-only mode (use HF Jobs for cloud GPU)"
fi

# 8. Build sandbox image
echo ""
echo "--- Building Sandbox Image ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" ]; then
    echo "Building surprisal-gpu image (this may take a few minutes)..."
    docker build -t surprisal-gpu:latest -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" "$SCRIPT_DIR/sandbox/"
else
    echo "Dockerfile.gpu not found — skipping image build"
    echo "Build later: docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/"
fi

# 9. Credentials (optional)
echo ""
echo "--- Credentials (optional, press Enter to skip) ---"
read -rp "W&B API key: " WANDB_KEY
read -rp "HuggingFace token: " HF_TOKEN

# 10. Write config
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/surprisal"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.toml" << EOF
[sandbox]
backend = "auto"
gpu = $GPU

[credentials]
wandb_api_key = "${WANDB_KEY:-}"
hf_token = "${HF_TOKEN:-}"
EOF
echo ""
echo "Config written to $CONFIG_DIR/config.toml"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  uv run surprisal init --domain 'your research topic' --seed 'your hypothesis'"
echo "  uv run surprisal explore --budget 10"
