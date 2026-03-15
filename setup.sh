#!/usr/bin/env bash
# Setup script for Terraria RL Agent — Kubuntu 25.10
set -e

VENV_DIR="$(dirname "$0")/.venv"
REQUIREMENTS="$(dirname "$0")/terraria_rl/requirements.txt"

echo "=== Terraria RL Agent — Python Setup ==="
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev

echo "      Python: $(python3 --version)"
echo ""

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[2/4] Virtual environment already exists at $VENV_DIR — skipping creation."
else
    echo "[2/4] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
echo ""

# ── 3. Install Python packages ────────────────────────────────────────────────
echo "[3/4] Installing Python packages..."
"$VENV_DIR/bin/pip" install --upgrade pip wheel

# PyTorch: try CUDA first, fall back to CPU-only if no GPU is found
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "      NVIDIA GPU detected — installing PyTorch with CUDA support."
    "$VENV_DIR/bin/pip" install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "      No NVIDIA GPU detected — installing CPU-only PyTorch."
    "$VENV_DIR/bin/pip" install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

# Remaining dependencies
"$VENV_DIR/bin/pip" install \
    "gymnasium>=0.29.0" \
    "tensorboard>=2.14.0" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0" \
    "flask>=3.0.0" \
    "flask-socketio>=5.3.0" \
    "eventlet>=0.35.0"

echo ""

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo "[4/4] Running smoke test..."
"$VENV_DIR/bin/python" - <<'EOF'
import torch, gymnasium, numpy, yaml, flask, flask_socketio
print(f"  torch       {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  gymnasium   {gymnasium.__version__}")
print(f"  numpy       {numpy.__version__}")
print(f"  flask       {flask.__version__}")
print("  All imports OK.")
EOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training:"
echo "  source .venv/bin/activate"
echo "  python terraria_rl/main.py --config terraria_rl/configs/default.yaml --mode train"
echo ""
echo "Web dashboard will be available at: http://localhost:5555"
