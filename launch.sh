#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  BossML Agent — Interactive Training Console
#  Structured like Claude Code: banner → status → REPL prompt
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/terraria_boss_agent"
VENV_PIP="$AGENT_DIR/.venv/bin/pip"
VENV_PYTHON="$AGENT_DIR/.venv/bin/python"
CONFIG="$AGENT_DIR/config/default.yaml"
PRESETS_DIR="$AGENT_DIR/config/presets"
ACTIVE_PRESET_FILE="$PRESETS_DIR/.active_preset"

# Accumulates --set flags for the training run
declare -a OVERRIDES=()
TRAIN_PID=""

# ── Colors & Formatting ───────────────────────────────────────
C_ORANGE='\033[38;5;209m'
C_DIM='\033[2m'
C_BOLD='\033[1m'
C_WHITE='\033[37m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_CYAN='\033[36m'
C_RED='\033[31m'
C_RESET='\033[0m'

# ── Clawd ─────────────────────────────────────────────────────
print_banner() {
    clear
    echo ""
    echo -e "  ${C_ORANGE}   ╱▔▔▔▔▔▔▔╲${C_RESET}"
    echo -e "  ${C_ORANGE}  │  ${C_WHITE}●   ●${C_ORANGE}  │${C_RESET}"
    echo -e "  ${C_ORANGE}  │   ${C_WHITE}▽${C_ORANGE}    │${C_RESET}"
    echo -e "  ${C_ORANGE}   ╲_______╱${C_RESET}"
    echo -e "  ${C_ORANGE}   ╱│${C_RESET}     ${C_ORANGE}│╲${C_RESET}"
    echo -e "  ${C_ORANGE}  ╱ │${C_RESET}     ${C_ORANGE}│ ╲${C_RESET}"
    echo -e "  ${C_ORANGE}    │${C_RESET}     ${C_ORANGE}│${C_RESET}"
    echo -e "  ${C_ORANGE}    ╱╲${C_RESET}   ${C_ORANGE}╱╲${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_ORANGE}BossML Agent${C_RESET}  ${C_DIM}v0.1.0 — Terraria RL Training Console${C_RESET}"
    echo -e "  ${C_DIM}Type ${C_WHITE}/help${C_DIM} to see available commands${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}────────────────────────────────────────────────${C_RESET}"
    echo ""
}

# ── Help ──────────────────────────────────────────────────────
print_help() {
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Commands${C_RESET}"
    echo ""
    echo -e "  ${C_CYAN}/train${C_RESET}                       Resume training (auto-loads latest model)"
    echo -e "  ${C_CYAN}/train new${C_RESET}                   Start fresh model from scratch"
    echo -e "  ${C_CYAN}/eval ${C_DIM}<model.zip>${C_RESET}            Evaluate a saved model"
    echo -e "  ${C_CYAN}/stop${C_RESET}                        Stop running training"
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Configuration${C_RESET}"
    echo ""
    echo -e "  ${C_CYAN}/weights${C_RESET}                     List all configurable weights"
    echo -e "  ${C_CYAN}/set ${C_DIM}<key>=<value>${C_RESET}           Override a config value"
    echo -e "  ${C_CYAN}/unset ${C_DIM}<key>${C_RESET}                Remove an override"
    echo -e "  ${C_CYAN}/show${C_RESET}                        Show current overrides"
    echo -e "  ${C_CYAN}/presets${C_RESET}                     List available presets"
    echo -e "  ${C_CYAN}/load ${C_DIM}<preset>${C_RESET}               Load a preset configuration"
    echo -e "  ${C_CYAN}/save ${C_DIM}<name>${C_RESET}                 Save current overrides as a preset"
    echo -e "  ${C_CYAN}/reset${C_RESET}                       Clear all overrides"
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Other${C_RESET}"
    echo ""
    echo -e "  ${C_CYAN}/status${C_RESET}                      Show training process status"
    echo -e "  ${C_CYAN}/help${C_RESET}                        Show this help"
    echo -e "  ${C_CYAN}/quit${C_RESET}                        Exit"
    echo ""
}

# ── Weights listing ───────────────────────────────────────────
get_weight_val() {
    local key="$1"
    local default="$2"
    for o in "${OVERRIDES[@]}"; do
        if [[ "${o%%=*}" == "$key" ]]; then
            echo "${o#*=}"
            return
        fi
    done
    echo "$default"
}

print_weight_line() {
    local key="$1"
    local default="$2"
    local desc="$3"
    local pad="$4"
    local val
    val=$(get_weight_val "$key" "$default")

    local yaml_val
    yaml_val=$("$VENV_PYTHON" -c "
import yaml
try:
    with open('$CONFIG', 'r') as f:
        cfg = yaml.safe_load(f) or {}
    parts = '$key'.split('.')
    d = cfg
    for p in parts[:-1]:
        d = d.get(p, {})
    file_val = d.get(parts[-1])
    if file_val is not None:
        print(str(file_val))
    else:
        print('')
except:
    print('')
" 2>/dev/null)

    if [[ -z "$yaml_val" ]]; then
        yaml_val="$default"
    fi

    if [[ "$val" != "$default" ]]; then
        echo -e "  ${C_CYAN}${key}${C_RESET}${pad}${C_YELLOW}[${val}]${C_RESET} ${C_DIM}(file: ${yaml_val}, default: ${default})${C_RESET} ${desc}"
    elif [[ "$yaml_val" != "$default" ]]; then
        echo -e "  ${C_CYAN}${key}${C_RESET}${pad}${C_YELLOW}[${yaml_val}]${C_RESET} ${C_DIM}(default: ${default})${C_RESET} ${desc}"
    else
        echo -e "  ${C_CYAN}${key}${C_RESET}${pad}${C_DIM}[${default}]${C_RESET} ${desc}"
    fi
}

print_weights() {
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Reward Weights${C_RESET}  ${C_DIM}(reward.*)${C_RESET}"
    echo ""
    print_weight_line "reward.boss_damage_weight"         "4"        "Reward per % boss HP dealt"                      "     "
    print_weight_line "reward.player_damage_weight"       "100.0"    "Penalty per % player HP lost"                    "   "
    print_weight_line "reward.survival_bonus"             "0.005"    "Bonus each step for staying alive"               "         "
    print_weight_line "reward.distance_penalty_weight"    "0.0005"   "Penalty for being far from boss"                 ""
    print_weight_line "reward.distance_threshold"         "240"      "Distance before penalty kicks in"                "     "
    print_weight_line "reward.boss_kill_bonus"            "100"      "Big reward for killing the boss"                 "        "
    print_weight_line "reward.death_penalty"              "100"      "Big penalty for dying"                           "          "
    print_weight_line "reward.wasteful_heal_penalty"      "1"        "Penalty for healing above 60% HP"                "  "
    print_weight_line "reward.despawn_penalty"            "100"      "Penalty when boss despawns"                      "        "
    print_weight_line "reward.distance_sq_penalty_weight" "5.0e-07"  "Quadratic distance penalty"                      ""
    print_weight_line "reward.proximity_radius_blocks"    "15"       "Bonus zone around boss (blocks)"                 " "
    print_weight_line "reward.proximity_bonus"            "0.5"      "Reward per step near boss"                       "        "
    print_weight_line "reward.dps_bonus_weight"           "1.0"      "DPS compounding multiplier"                      "       "
    print_weight_line "reward.idle_penalty_per_sec"       "1.0"      "Idle penalty (compounds each sec)"               "   "
    print_weight_line "reward.flawless_kill_bonus"        "200.0"    "Kill bonus scaled by 1/(1+hits)"                 "    "
    print_weight_line "reward.dodge_streak_bonus"         "0.0"      "Dodge streak bonus (compounds)"                  "    "
    print_weight_line "reward.dodge_streak_radius_blocks" "15"       "Dodge streak radius (blocks)"                    ""
    print_weight_line "reward.nohit_bonus"                "20"       "Incremental nohit bonus (per-step + kill)"       "           "
    print_weight_line "reward.hit_compounding_penalty"    "5"        "Per-hit compounding penalty (Nth hit = N*w)"     "  "
    print_weight_line "reward.nohit_kill_bonus"           "10000"    "Massive bonus for killing boss without any hits" "      "
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}PPO Hyperparameters${C_RESET}  ${C_DIM}(training.*)${C_RESET}"
    echo ""
    print_weight_line "training.learning_rate"   "0.00015"   "Adam learning rate"          "        "
    print_weight_line "training.gamma"           "0.997"     "Discount factor"             "                "
    print_weight_line "training.gae_lambda"      "0.98"      "GAE lambda"                  "           "
    print_weight_line "training.clip_range"      "0.15"      "PPO clip range"              "           "
    print_weight_line "training.ent_coef"        "0.01"      "Entropy bonus (exploration)" "            "
    print_weight_line "training.vf_coef"         "0.5"       "Value function loss coeff"   "              "
    print_weight_line "training.max_grad_norm"   "0.5"       "Gradient clipping"           "        "
    print_weight_line "training.n_steps"         "40960"     "Steps per rollout"           "              "
    print_weight_line "training.batch_size"      "1280"      "Minibatch size"              "           "
    print_weight_line "training.n_epochs"        "8"         "PPO epochs per rollout"      "            "
    print_weight_line "training.total_timesteps" "10000000"  "Total training steps"        "      "
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Episode${C_RESET}  ${C_DIM}(episode.*)${C_RESET}"
    echo ""
    print_weight_line "episode.max_steps"    "54000" "Max steps per episode (~15 min)" "             "
    print_weight_line "episode.reward_scale" "1.0"   "Global reward multiplier"        "          "
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Boss${C_RESET}  ${C_DIM}(top-level)${C_RESET}"
    echo ""
    print_weight_line "boss_type" "50" "NPC type ID (50=KS 4=EoC 35=Skel 222=QB)" "                     "
    echo ""
}

# ── Presets ────────────────────────────────────────────────────
ensure_presets() {
    mkdir -p "$PRESETS_DIR"

    # ── clean: simple reward signal, correct entropy, good baseline ──
    if [ ! -f "$PRESETS_DIR/clean.conf" ]; then
        cat > "$PRESETS_DIR/clean.conf" << 'PRESET'
# Clean: simple reward signal — good starting point for any boss
training.ent_coef=0.01
training.learning_rate=0.00015
reward.boss_damage_weight=4
reward.player_damage_weight=5
reward.boss_kill_bonus=500
reward.death_penalty=200
reward.survival_bonus=0.01
reward.despawn_penalty=100
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi

    # ── nohit: use AFTER model can win consistently ──
    if [ ! -f "$PRESETS_DIR/nohit.conf" ]; then
        cat > "$PRESETS_DIR/nohit.conf" << 'PRESET'
# NoHit: use only after model wins reliably with clean preset
# Gradually pushes toward damage-free runs
training.ent_coef=0.005
training.learning_rate=0.00010
reward.boss_damage_weight=4
reward.player_damage_weight=20
reward.boss_kill_bonus=500
reward.death_penalty=300
reward.survival_bonus=0.01
reward.despawn_penalty=100
reward.nohit_bonus=5
reward.nohit_kill_bonus=2000
reward.flawless_kill_bonus=500
reward.hit_compounding_penalty=10
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi

    # ── aggressive: maximize damage, tolerate hits ──
    if [ ! -f "$PRESETS_DIR/aggressive.conf" ]; then
        cat > "$PRESETS_DIR/aggressive.conf" << 'PRESET'
# Aggressive: prioritize dealing damage over survival
training.ent_coef=0.01
reward.boss_damage_weight=10
reward.player_damage_weight=1
reward.boss_kill_bonus=500
reward.death_penalty=100
reward.survival_bonus=0.005
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi

    # ── tank: survive at all costs ──
    if [ ! -f "$PRESETS_DIR/tank.conf" ]; then
        cat > "$PRESETS_DIR/tank.conf" << 'PRESET'
# Tank: prioritize survival, careful play
training.ent_coef=0.01
reward.boss_damage_weight=2
reward.player_damage_weight=15
reward.death_penalty=400
reward.boss_kill_bonus=500
reward.survival_bonus=0.02
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi

    # ── speedrun: kill fast ──
    if [ ! -f "$PRESETS_DIR/speedrun.conf" ]; then
        cat > "$PRESETS_DIR/speedrun.conf" << 'PRESET'
# Speedrun: kill the boss as fast as possible
training.ent_coef=0.01
reward.boss_damage_weight=8
reward.player_damage_weight=2
reward.boss_kill_bonus=1000
reward.death_penalty=200
reward.survival_bonus=-0.005
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
episode.max_steps=9000
PRESET
    fi

    # ── explore: high entropy for early experimentation ──
    if [ ! -f "$PRESETS_DIR/explore.conf" ]; then
        cat > "$PRESETS_DIR/explore.conf" << 'PRESET'
# Explore: higher entropy for discovering strategies early
training.ent_coef=0.03
training.learning_rate=0.0001
reward.boss_damage_weight=4
reward.player_damage_weight=5
reward.boss_kill_bonus=500
reward.death_penalty=200
reward.survival_bonus=0.01
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi

    # ── quick: fast iteration for testing ──
    if [ ! -f "$PRESETS_DIR/quick.conf" ]; then
        cat > "$PRESETS_DIR/quick.conf" << 'PRESET'
# Quick: short training run for testing config changes
training.ent_coef=0.01
training.total_timesteps=200000
training.n_steps=4096
training.batch_size=128
reward.boss_damage_weight=4
reward.player_damage_weight=5
reward.boss_kill_bonus=500
reward.death_penalty=200
reward.survival_bonus=0.01
reward.nohit_bonus=0
reward.nohit_kill_bonus=0
reward.flawless_kill_bonus=0
reward.hit_compounding_penalty=0
reward.proximity_bonus=0
reward.distance_penalty_weight=0
reward.distance_sq_penalty_weight=0
reward.wasteful_heal_penalty=0
reward.dodge_streak_bonus=0
reward.dps_bonus_weight=0
reward.idle_penalty_per_sec=0
PRESET
    fi
}

list_presets() {
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Available Presets${C_RESET}"
    echo ""
    for f in "$PRESETS_DIR"/*.conf; do
        [ -f "$f" ] || continue
        local name
        name=$(basename "$f" .conf)
        local desc
        desc=$(grep '^#' "$f" | head -1 | sed 's/^# *//')
        echo -e "  ${C_CYAN}${name}${C_RESET}  ${C_DIM}— ${desc}${C_RESET}"
    done
    echo ""
    echo -e "  ${C_DIM}Recommended order: clean → nohit${C_RESET}"
    echo -e "  ${C_DIM}Usage: /load clean${C_RESET}"
    echo ""
}

load_preset() {
    local name="$1"
    local file="$PRESETS_DIR/${name}.conf"
    if [ ! -f "$file" ]; then
        echo -e "  ${C_RED}Preset '${name}' not found.${C_RESET} Use ${C_CYAN}/presets${C_RESET} to list."
        return 1
    fi

    OVERRIDES=()

    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue
        OVERRIDES+=("$line")
    done < "$file"

    echo "$name" > "$ACTIVE_PRESET_FILE"

    echo -e "  ${C_GREEN}Loaded preset '${name}'${C_RESET} (${#OVERRIDES[@]} overrides)"
    show_overrides

    apply_overrides_to_yaml
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo -e "  ${C_DIM}Applied to running training (hot-reload on next episode)${C_RESET}"
    fi
}

save_preset() {
    local name="$1"
    local file="$PRESETS_DIR/${name}.conf"

    if [ ${#OVERRIDES[@]} -eq 0 ]; then
        echo -e "  ${C_YELLOW}No overrides to save.${C_RESET} Use ${C_CYAN}/set key=value${C_RESET} first."
        return
    fi

    mkdir -p "$PRESETS_DIR"
    {
        echo "# Preset: ${name}"
        echo "# Saved $(date '+%Y-%m-%d %H:%M')"
        for o in "${OVERRIDES[@]}"; do
            echo "$o"
        done
    } > "$file"

    echo -e "  ${C_GREEN}Saved preset '${name}'${C_RESET} (${#OVERRIDES[@]} overrides) to ${C_DIM}${file}${C_RESET}"
}

# ── Override management ───────────────────────────────────────
show_overrides() {
    if [ ${#OVERRIDES[@]} -eq 0 ]; then
        echo -e "  ${C_DIM}No overrides set. Using defaults from config/default.yaml${C_RESET}"
        return
    fi
    echo ""
    echo -e "  ${C_BOLD}${C_WHITE}Active Overrides${C_RESET}"
    echo ""
    for o in "${OVERRIDES[@]}"; do
        local key="${o%%=*}"
        local val="${o#*=}"
        echo -e "  ${C_CYAN}${key}${C_RESET} = ${C_YELLOW}${val}${C_RESET}"
    done
    echo ""
}

set_override() {
    local kv="$1"
    if [[ ! "$kv" == *"="* ]]; then
        echo -e "  ${C_RED}Usage: /set key=value${C_RESET}"
        return
    fi
    local key="${kv%%=*}"

    local -a new_overrides=()
    for o in "${OVERRIDES[@]}"; do
        [[ "${o%%=*}" != "$key" ]] && new_overrides+=("$o")
    done
    new_overrides+=("$kv")
    OVERRIDES=("${new_overrides[@]}")

    echo -e "  ${C_GREEN}Set${C_RESET} ${C_CYAN}${key}${C_RESET} = ${C_YELLOW}${kv#*=}${C_RESET}"
    sync_if_running
}

unset_override() {
    local key="$1"
    local -a new_overrides=()
    local found=false
    for o in "${OVERRIDES[@]}"; do
        if [[ "${o%%=*}" == "$key" ]]; then
            found=true
        else
            new_overrides+=("$o")
        fi
    done
    OVERRIDES=("${new_overrides[@]}")

    if $found; then
        echo -e "  ${C_GREEN}Removed override${C_RESET} ${C_CYAN}${key}${C_RESET}"
        sync_if_running
    else
        echo -e "  ${C_DIM}No override found for '${key}'${C_RESET}"
    fi
}

# ── Live config sync ──────────────────────────────────────────
apply_overrides_to_yaml() {
    if [ ${#OVERRIDES[@]} -eq 0 ]; then
        return
    fi

    local -a py_args=()
    for o in "${OVERRIDES[@]}"; do
        py_args+=("$o")
    done

    "$VENV_PYTHON" -c "
import sys, yaml

config_path = '$CONFIG'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f) or {}

for arg in sys.argv[1:]:
    key, val = arg.split('=', 1)
    parts = key.split('.')
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    if val.lower() in ('true', 'false'):
        d[parts[-1]] = val.lower() == 'true'
    else:
        try:
            d[parts[-1]] = int(val)
        except ValueError:
            try:
                d[parts[-1]] = float(val)
            except ValueError:
                d[parts[-1]] = val

with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "${py_args[@]}"
}

sync_if_running() {
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        apply_overrides_to_yaml
        echo -e "  ${C_DIM}Applied to running training (hot-reload on next episode)${C_RESET}"
    fi
}

# ── Training control ──────────────────────────────────────────
start_training() {
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo -e "  ${C_YELLOW}Training already running (PID $TRAIN_PID).${C_RESET} Use ${C_CYAN}/stop${C_RESET} first."
        return
    fi

    local -a set_args=()
    for o in "${OVERRIDES[@]}"; do
        set_args+=("--set" "$o")
    done

    echo ""
    echo -e "  ${C_ORANGE}(◕ ‿ ◕)${C_RESET} ${C_BOLD}Starting training...${C_RESET}"
    if [ ${#OVERRIDES[@]} -gt 0 ]; then
        echo -e "  ${C_DIM}With ${#OVERRIDES[@]} override(s)${C_RESET}"
    fi
    echo ""

    cd "$AGENT_DIR"
    local -a extra_args=()
    if [ "${1:-}" = "new" ]; then
        extra_args+=("--new")
        echo -e "  ${C_DIM}Starting fresh model (ignoring existing checkpoints)${C_RESET}"
    fi
    "$VENV_PYTHON" -m src.training.train --config config/default.yaml "${extra_args[@]}" "${set_args[@]}" &
    TRAIN_PID=$!
    echo -e "  ${C_GREEN}Training started${C_RESET} ${C_DIM}(PID $TRAIN_PID)${C_RESET}"
}

start_eval() {
    local model_path="$1"
    if [ ! -f "$model_path" ] && [ ! -f "$AGENT_DIR/$model_path" ]; then
        echo -e "  ${C_RED}Model not found: ${model_path}${C_RESET}"
        return
    fi
    echo -e "  ${C_ORANGE}(● ◡ ●)${C_RESET} ${C_BOLD}Evaluating model: ${model_path}${C_RESET}"
    cd "$AGENT_DIR"
    "$VENV_PYTHON" -m src.eval.evaluate --config config/default.yaml --model "$model_path" &
    TRAIN_PID=$!
}

stop_training() {
    if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo -e "  ${C_DIM}No training process running.${C_RESET}"
        TRAIN_PID=""
        return
    fi
    kill "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
    echo -e "  ${C_YELLOW}Training stopped.${C_RESET}"
    TRAIN_PID=""
}

show_status() {
    echo ""
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo -e "  ${C_GREEN}●${C_RESET} Training ${C_BOLD}running${C_RESET} ${C_DIM}(PID $TRAIN_PID)${C_RESET}"
    else
        echo -e "  ${C_DIM}●${C_RESET} Training ${C_DIM}not running${C_RESET}"
        TRAIN_PID=""
    fi
    echo -e "  ${C_DIM}Config: ${CONFIG}${C_RESET}"
    echo -e "  ${C_DIM}Overrides: ${#OVERRIDES[@]}${C_RESET}"
    echo ""
}

# ── Cleanup ───────────────────────────────────────────────────
cleanup() {
    echo ""
    stop_training
    echo -e "  ${C_ORANGE}(● _ ●)${C_RESET} ${C_DIM}Bye!${C_RESET}"
    echo ""
    exit 0
}
trap cleanup INT TERM

# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════

print_banner

# ── Ensure venv ───────────────────────────────────────────────
if [ ! -d "$AGENT_DIR/.venv" ]; then
    echo -e "  ${C_DIM}Creating Python venv...${C_RESET}"
    python3 -m venv "$AGENT_DIR/.venv"
    echo -e "  ${C_DIM}Installing dependencies...${C_RESET}"
    "$VENV_PIP" install --quiet -e "$AGENT_DIR"
fi

"$VENV_PYTHON" -c "import gymnasium; import stable_baselines3; import torch; import yaml" 2>/dev/null || {
    echo -e "  ${C_YELLOW}Installing missing packages...${C_RESET}"
    "$VENV_PIP" install --quiet -e "$AGENT_DIR"
}

echo -e "  ${C_GREEN}✓${C_RESET} Python environment ready"

# ── Ensure presets exist ──────────────────────────────────────
ensure_presets
echo -e "  ${C_GREEN}✓${C_RESET} ${C_DIM}$(ls "$PRESETS_DIR"/*.conf 2>/dev/null | wc -l) presets available${C_RESET}"

# ── Auto-restore last preset ──────────────────────────────────
# NOTE: only restores if the preset file still exists
# If you want to start clean, run /reset before /train
if [ -f "$ACTIVE_PRESET_FILE" ]; then
    _active=$(cat "$ACTIVE_PRESET_FILE")
    if [ -f "$PRESETS_DIR/${_active}.conf" ]; then
        OVERRIDES=()
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// /}" ]] && continue
            OVERRIDES+=("$line")
        done < "$PRESETS_DIR/${_active}.conf"
        echo -e "  ${C_GREEN}✓${C_RESET} ${C_DIM}Preset '${_active}' restored (${#OVERRIDES[@]} overrides)${C_RESET}"
    fi
fi

echo ""
echo -e "  ${C_DIM}────────────────────────────────────────────────${C_RESET}"
echo ""

# ── Handle CLI args (non-interactive mode) ────────────────────
if [ $# -gt 0 ]; then
    case "$1" in
        --eval)
            shift
            start_eval "$1"
            wait "$TRAIN_PID" 2>/dev/null
            exit $?
            ;;
        --train)
            shift
            while [ $# -gt 0 ]; do
                case "$1" in
                    --set) shift; OVERRIDES+=("$1") ;;
                    *) ;;
                esac
                shift
            done
            start_training
            wait "$TRAIN_PID" 2>/dev/null
            exit $?
            ;;
    esac
fi

# ═══════════════════════════════════════════════════════════════
#  REPL
# ═══════════════════════════════════════════════════════════════

while true; do
    echo -ne "${C_ORANGE}❯${C_RESET} "
    read -r cmd args || break

    case "$cmd" in
        /help|help)
            print_help
            ;;
        /train|train)
            start_training "$args"
            ;;
        /eval|eval)
            if [ -z "$args" ]; then
                echo -e "  ${C_RED}Usage: /eval <path/to/model.zip>${C_RESET}"
            else
                start_eval "$args"
            fi
            ;;
        /stop|stop)
            stop_training
            ;;
        /status|status)
            show_status
            ;;
        /weights|weights)
            print_weights
            ;;
        /set|set)
            if [ -z "$args" ]; then
                echo -e "  ${C_RED}Usage: /set reward.boss_damage_weight=2.0${C_RESET}"
            else
                set_override "$args"
            fi
            ;;
        /unset|unset)
            if [ -z "$args" ]; then
                echo -e "  ${C_RED}Usage: /unset reward.boss_damage_weight${C_RESET}"
            else
                unset_override "$args"
            fi
            ;;
        /show|show)
            show_overrides
            ;;
        /presets|presets)
            list_presets
            ;;
        /load|load)
            if [ -z "$args" ]; then
                echo -e "  ${C_RED}Usage: /load clean${C_RESET}"
                list_presets
            else
                load_preset "$args"
            fi
            ;;
        /save|save)
            if [ -z "$args" ]; then
                echo -e "  ${C_RED}Usage: /save mypreset${C_RESET}"
            else
                save_preset "$args"
            fi
            ;;
        /reset)
            OVERRIDES=()
            rm -f "$ACTIVE_PRESET_FILE"
            echo -e "  ${C_GREEN}All overrides cleared.${C_RESET}"
            if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
                echo -e "  ${C_YELLOW}Note: running training still uses previous values until restarted.${C_RESET}"
            fi
            ;;
        /quit|/exit|quit|exit)
            cleanup
            ;;
        "")
            ;;
        *)
            echo -e "  ${C_DIM}Unknown command '${cmd}'. Type ${C_WHITE}/help${C_DIM} for commands.${C_RESET}"
            ;;
    esac
done