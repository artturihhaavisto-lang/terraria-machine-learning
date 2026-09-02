#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_terraria_ml.sh
#
#  One-shot setup script that recreates the full environment for:
#    https://github.com/artturihhaavisto-lang/terraria-machine-learning
#
#  A Terraria boss-fighting deep-RL framework consisting of:
#    • BossMLMod / TerrariaRLAgent  — tModLoader C# mods (TCP state server)
#    • terraria_boss_agent          — SB3/PPO Gymnasium agent (pip package)
#    • terraria_rl                  — custom PPO impl + Flask web dashboard
#
#  What this script does:
#    1. Installs system dependencies (apt-based distros; skipped elsewhere)
#    2. Clones the repository (branch: claude/terraria-rl-agent-P7oH6)
#    3. Creates a Python virtual environment (.venv)
#    4. Installs PyTorch (auto-detects CUDA GPU, falls back to CPU wheels)
#    5. Installs both Python packages (terraria_boss_agent + terraria_rl deps)
#    6. Symlinks the tModLoader mods into ModSources for in-game building
#    7. Runs a smoke test and prints next steps
#
#  Usage:
#    ./setup_terraria_ml.sh [options]
#
#  Options:
#    --dir <path>        Install directory (default: ./terraria-machine-learning)
#    --branch <name>     Git branch to check out (default: repo default branch)
#    --cpu               Force CPU-only PyTorch even if a GPU is present
#    --skip-system       Skip apt system-dependency installation (no sudo)
#    --skip-mod-link     Don't symlink mods into tModLoader ModSources
#    --python <bin>      Python interpreter to use (default: auto, prefers 3.11+)
#    -h, --help          Show this help
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_URL="https://github.com/artturihhaavisto-lang/terraria-machine-learning.git"
INSTALL_DIR="$(pwd)/terraria-machine-learning"
BRANCH=""                      # empty = repo default branch
FORCE_CPU=0
SKIP_SYSTEM=0
SKIP_MOD_LINK=0
PYTHON_BIN=""
MODSOURCES="${MODSOURCES:-$HOME/.local/share/Terraria/tModLoader/ModSources}"

# ── Pretty printing ──────────────────────────────────────────────────────────
BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; YELLOW='\033[33m'
RED='\033[31m'; CYAN='\033[36m'; RESET='\033[0m'
step()  { echo -e "\n${BOLD}${CYAN}==>${RESET}${BOLD} $*${RESET}"; }
info()  { echo -e "    ${DIM}$*${RESET}"; }
ok()    { echo -e "    ${GREEN}✔${RESET} $*"; }
warn()  { echo -e "    ${YELLOW}⚠${RESET} $*"; }
fail()  { echo -e "    ${RED}✘ $*${RESET}"; exit 1; }

usage() { sed -n '2,32p' "$0" | sed 's/^#[ ]\{0,2\}//'; exit 0; }

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)           INSTALL_DIR="$(realpath -m "$2")"; shift 2 ;;
        --branch)        BRANCH="$2"; shift 2 ;;
        --cpu)           FORCE_CPU=1; shift ;;
        --skip-system)   SKIP_SYSTEM=1; shift ;;
        --skip-mod-link) SKIP_MOD_LINK=1; shift ;;
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *)               fail "Unknown option: $1 (see --help)" ;;
    esac
done

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║   Terraria Machine Learning — Environment Setup          ║"
echo "  ║   tModLoader mod + PPO reinforcement-learning agent      ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "Install dir : $INSTALL_DIR"
info "ModSources  : $MODSOURCES"

# ── 1. System dependencies ───────────────────────────────────────────────────
step "[1/7] System dependencies"
if [[ $SKIP_SYSTEM -eq 1 ]]; then
    info "Skipped (--skip-system)."
elif command -v apt-get &>/dev/null; then
    SUDO=""
    [[ $EUID -ne 0 ]] && SUDO="sudo"
    info "Detected apt-based system — installing packages (may prompt for sudo)..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        git python3 python3-venv python3-pip python3-dev \
        build-essential libssl-dev libffi-dev curl ca-certificates \
        xvfb
    # .NET SDK is optional — only needed for IDE code completion on the C# mod.
    if ! command -v dotnet &>/dev/null; then
        if $SUDO apt-get install -y -qq dotnet-sdk-8.0 2>/dev/null \
        || $SUDO apt-get install -y -qq dotnet-sdk-6.0 2>/dev/null; then
            ok ".NET SDK installed (for C# mod IDE support)."
        else
            warn ".NET SDK not available via apt — skipping (only needed for IDE completion; the mod is built in-game by tModLoader)."
        fi
    fi
    ok "System packages installed."
else
    warn "Non-apt system detected — install git, python3 (3.11+), venv, pip and build tools manually."
fi

command -v git &>/dev/null || fail "git is required but not found."

# ── 2. Pick a Python interpreter ─────────────────────────────────────────────
step "[2/7] Locating Python (3.11+ preferred)"
if [[ -z "$PYTHON_BIN" ]]; then
    for cand in python3.13 python3.12 python3.11 python3; do
        if command -v "$cand" &>/dev/null; then PYTHON_BIN="$cand"; break; fi
    done
fi
[[ -n "$PYTHON_BIN" ]] || fail "No python3 interpreter found."
PYVER="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "Python 3.11+ required, found $PYVER ($PYTHON_BIN). Use --python to point at a newer interpreter."
ok "Using $PYTHON_BIN (Python $PYVER)"

# ── 3. Clone the repository ──────────────────────────────────────────────────
step "[3/7] Cloning repository"
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repository already exists — pulling latest changes."
    git -C "$INSTALL_DIR" pull --ff-only || warn "Could not fast-forward; continuing with existing checkout."
else
    CLONE_ARGS=(--depth 1)
    [[ -n "$BRANCH" ]] && CLONE_ARGS+=(--branch "$BRANCH")
    git clone "${CLONE_ARGS[@]}" "$REPO_URL" "$INSTALL_DIR"
fi
ok "Repository ready at $INSTALL_DIR ($(git -C "$INSTALL_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$INSTALL_DIR" rev-parse --short HEAD))"

VENV_DIR="$INSTALL_DIR/.venv"
PIP="$VENV_DIR/bin/pip"
VPY="$VENV_DIR/bin/python"

# ── 4. Virtual environment ───────────────────────────────────────────────────
step "[4/7] Creating virtual environment"
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists — reusing."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$PIP" install --quiet --upgrade pip wheel setuptools
ok "venv ready at $VENV_DIR"

# ── 5. PyTorch (CUDA auto-detect) ────────────────────────────────────────────
step "[5/7] Installing PyTorch"
if [[ $FORCE_CPU -eq 0 ]] && command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    info "NVIDIA GPU detected — installing CUDA build (cu124)."
    "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    info "No usable NVIDIA GPU (or --cpu given) — installing CPU-only build."
    "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
ok "PyTorch installed."

# ── 6. Python packages ───────────────────────────────────────────────────────
step "[6/7] Installing Python packages"

# 6a. terraria_boss_agent — SB3-based agent, installed as an editable package
info "terraria_boss_agent (Stable-Baselines3 PPO agent)..."
"$PIP" install -e "$INSTALL_DIR/terraria_boss_agent"

# 6b. terraria_rl — custom PPO + Flask/SocketIO web dashboard dependencies
info "terraria_rl (custom PPO + web dashboard)..."
if [[ -f "$INSTALL_DIR/terraria_rl/requirements.txt" ]]; then
    "$PIP" install -r "$INSTALL_DIR/terraria_rl/requirements.txt"
fi
"$PIP" install "flask-socketio>=5.3.0" "eventlet>=0.35.0"

chmod +x "$INSTALL_DIR"/launch.sh "$INSTALL_DIR"/setup.sh \
         "$INSTALL_DIR"/terraria_boss_agent/scripts/*.sh 2>/dev/null || true

# 6c. Multi-instance extras — instance manager + parallel trainer.
#     These ship alongside this setup script; copy them into the project.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/tml-instances.sh" ]]; then
    install -m 0755 "$SCRIPT_DIR/tml-instances.sh" "$INSTALL_DIR/tml-instances.sh"
    ok "Installed tml-instances.sh (headless multi-instance manager)."
else
    warn "tml-instances.sh not found next to this script — multi-instance manager not installed."
fi
if [[ -f "$SCRIPT_DIR/train_parallel.py" ]]; then
    install -m 0755 "$SCRIPT_DIR/train_parallel.py" \
        "$INSTALL_DIR/terraria_boss_agent/scripts/train_parallel.py"
    ok "Installed scripts/train_parallel.py (vectorized PPO trainer)."
else
    warn "train_parallel.py not found next to this script — parallel trainer not installed."
fi
if [[ -f "$SCRIPT_DIR/fleet.sh" ]]; then
    install -m 0755 "$SCRIPT_DIR/fleet.sh" "$INSTALL_DIR/fleet.sh"
    ok "Installed fleet.sh (single-terminal fleet console)."
else
    warn "fleet.sh not found next to this script — fleet console not installed."
fi
if [[ -f "$SCRIPT_DIR/boss_profiles.conf" && ! -f "$INSTALL_DIR/boss_profiles.conf" ]]; then
    install -m 0644 "$SCRIPT_DIR/boss_profiles.conf" "$INSTALL_DIR/boss_profiles.conf"
    ok "Installed boss_profiles.conf (per-boss arena/time/world profiles)."
fi

# 6d. Patch BossMLMod source for headless automation (AutoEnableML,
#     ForceTimeOfDay). Idempotent; mod must be rebuilt in-game once after.
if [[ -f "$SCRIPT_DIR/patch_bossml_mod.py" ]]; then
    install -m 0755 "$SCRIPT_DIR/patch_bossml_mod.py" "$INSTALL_DIR/patch_bossml_mod.py"
    python3 "$INSTALL_DIR/patch_bossml_mod.py" "$INSTALL_DIR/BossMLMod"
else
    warn "patch_bossml_mod.py not found — headless automation (automl/time) unavailable."
fi

ok "All Python packages installed."

# ── 7. tModLoader mod sources + smoke test ───────────────────────────────────
step "[7/7] tModLoader mods & smoke test"
if [[ $SKIP_MOD_LINK -eq 1 ]]; then
    info "Mod symlinking skipped (--skip-mod-link)."
else
    mkdir -p "$MODSOURCES"
    for MOD in BossMLMod TerrariaRLAgent; do
        if [[ -e "$MODSOURCES/$MOD" ]]; then
            warn "$MODSOURCES/$MOD already exists — leaving as-is."
        else
            ln -s "$INSTALL_DIR/$MOD" "$MODSOURCES/$MOD"
            ok "Symlinked $MOD → ModSources"
        fi
    done
fi

info "Running import smoke test..."
"$VPY" - <<'EOF'
import torch, gymnasium, numpy, yaml, flask, flask_socketio, stable_baselines3
print(f"      torch             {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"      gymnasium         {gymnasium.__version__}")
print(f"      stable-baselines3 {stable_baselines3.__version__}")
print(f"      numpy             {numpy.__version__}")
print(f"      flask             {flask.__version__}")
print("      All imports OK.")
EOF
ok "Smoke test passed."

# ── Done — next steps ────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}═══ Setup complete! ═══${RESET}\n"
cat <<NEXT
Next steps (in order):

  1. Install tModLoader 1.4.4 via Steam:
       Terraria → Properties → Betas → select "tModLoader"
       (or install the standalone tModLoader app, Steam app ID 1281930)

  2. Build & enable the mod IN-GAME:
       Launch tModLoader → Workshop → Develop Mods → BossMLMod → Build
       Then: Mods menu → enable it → Reload

  3. Prepare a world:
       - Create/load any character + world
       - Build a flat arena (~200 tiles wide, campfire + heart lantern)
       - In-game chat:  /bml arena     (save arena center)
                        /bml boss 50   (King Slime; see QUICKSTART.md for IDs)
                        /bml status    (verify)
                        /bml on        (agent takes control)

  4. Start training (SB3 agent):
       cd "$INSTALL_DIR"
       source .venv/bin/activate
       ./terraria_boss_agent/scripts/run_training.sh

     — or the custom PPO stack with the web dashboard (http://localhost:5555):
       python terraria_rl/main.py --config terraria_rl/configs/default.yaml --mode train

     — or the interactive training console:
       ./launch.sh

  PARALLEL / HEADLESS TRAINING — single-terminal fleet console:

       # One-time: run tModLoader normally once (step 2-3 above) so your
       # save dir has a player, world, built BossMLMod and saved arena.

       cd "$INSTALL_DIR"
       ./fleet.sh                     # opens the console, then inside it:
         fleet> automl on             #   one-time: ML mode arms itself on connect
         fleet> boss 4                #   boss + profile (platforms/time/world)
         fleet> start 4               #   4 headless games, ports 7777+
         fleet> train                 #   parallel PPO on all ready instances
         fleet> tb                    #   TensorBoard at :6006
         fleet> watch                 #   live dashboard
         fleet> train stop / stop     #   wind down

     NOTE: automl/time need the patched BossMLMod — rebuild it once in
     tModLoader (Workshop → Develop Mods → BossMLMod → Build) after setup.
     Edit boss_profiles.conf to map your arena worlds to each boss stage.

     One-shot mode also works: ./fleet.sh status, ./fleet.sh stop
     (Lower level: ./tml-instances.sh + terraria_boss_agent/scripts/train_parallel.py)

     Tip: SB3's n_steps is per-env — with 4 envs consider
       train --set training.n_steps=1024 to keep the same rollout size.

  5. Monitor:
       tensorboard --logdir logs     # http://localhost:6006

Headless training (no monitor):
       sudo apt install xvfb
       Xvfb :99 -screen 0 1280x720x24 &
       DISPLAY=:99 steam -applaunch 1281930

Docs: $INSTALL_DIR/README.md, QUICKSTART.md, DESIGN.md
NEXT
