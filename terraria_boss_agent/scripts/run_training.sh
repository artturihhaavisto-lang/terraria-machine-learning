#!/usr/bin/env bash
# Train the Terraria boss-fighting agent.
#
# Prerequisites:
#   1. Terraria + tModLoader running with BossMLMod enabled
#   2. In-game: /bml arena  (set arena center)
#   3. In-game: /bml boss 50  (or whichever boss ID)
#   4. In-game: /bml on  (enable ML mode)
#
# Usage:
#   ./scripts/run_training.sh                          # defaults
#   ./scripts/run_training.sh --resume logs/models/terraria_ppo_50000_steps.zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== Terraria Boss Agent Training ==="
echo "Config: config/default.yaml"
echo "Connecting to localhost:7777..."
echo ""
echo "Make sure the game is running with BossMLMod and ML mode is ON (/bml on)"
echo ""

python -m src.training.train --config config/default.yaml "$@"
