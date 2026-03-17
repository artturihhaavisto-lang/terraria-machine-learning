#!/usr/bin/env bash
# Evaluate a trained agent (deterministic policy).
#
# Usage:
#   ./scripts/run_eval.sh --model logs/models/final_model.zip
#   ./scripts/run_eval.sh --model logs/models/final_model.zip --episodes 50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 --model <path_to_model.zip> [--episodes N] [--output results.json]"
    exit 1
fi

echo "=== Terraria Boss Agent Evaluation ==="
python -m src.eval.evaluate --config config/default.yaml "$@"
