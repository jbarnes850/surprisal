#!/usr/bin/env bash
set -euo pipefail

echo "=== Surprisal Setup ==="
echo ""

# 1. Check Python
if ! python3 --version >/dev/null 2>&1; then
    echo "ERROR: Python 3 required. Install from https://python.org"
    exit 1
fi
echo "Python: $(python3 --version)"

# 2. Check Docker
if ! docker --version >/dev/null 2>&1; then
    echo "ERROR: Docker required. Install from https://docs.docker.com/get-docker/"
    exit 1
fi
echo "Docker: $(docker --version)"

# 3. Install surprisal
echo ""
echo "--- Installing surprisal ---"
pip3 install surprisal

# 4. Check/setup Claude CLI
echo ""
echo "--- Agent Setup ---"
if ! command -v claude &>/dev/null; then
    echo "Installing Claude CLI..."
    npm install -g @anthropic-ai/claude-code
fi
if ! claude auth status 2>/dev/null | grep -q loggedIn; then
    echo "Please log in to Claude:"
    claude auth login
fi
echo "Claude CLI: ready"

if command -v codex &>/dev/null; then
    echo "Codex CLI: detected (optional)"
fi

# 5. Check GPU
echo ""
echo "--- GPU Detection ---"
GPU=false
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU detected: $GPU_NAME"
    GPU=true
else
    echo "No GPU detected -- stats-only mode (use HF Jobs for cloud GPU)"
fi

# 6. Build sandbox image
echo ""
echo "--- Building Sandbox Image ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$GPU" = true ] && [ -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" ]; then
    docker build -t surprisal-gpu:latest -f "$SCRIPT_DIR/sandbox/Dockerfile.gpu" "$SCRIPT_DIR/sandbox/"
elif [ -f "$SCRIPT_DIR/sandbox/Dockerfile" ]; then
    docker build -t surprisal-sandbox:latest "$SCRIPT_DIR/sandbox/"
else
    echo "Sandbox Dockerfiles not found -- skipping image build"
    echo "You can build later with: docker build -t surprisal-gpu:latest -f sandbox/Dockerfile.gpu sandbox/"
fi

# 7. Credentials (optional)
echo ""
echo "--- Credentials (optional, press Enter to skip) ---"
read -rp "W&B API key: " WANDB_KEY
read -rp "HuggingFace token: " HF_TOKEN

# 8. Write config
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
echo "  surprisal init --domain 'your research topic' --seed 'your hypothesis'"
echo "  surprisal explore --budget 10"
