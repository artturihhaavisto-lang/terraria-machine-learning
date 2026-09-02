#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  fleet.sh — Single-terminal command console for the whole Terraria RL fleet
#
#  One dashboard controls everything: headless game instances, the parallel
#  PPO trainer, TensorBoard, and fleet-wide BossMLMod configuration.
#
#  Usage:  ./fleet.sh            interactive console
#          ./fleet.sh <command>  run one command and exit (e.g. ./fleet.sh status)
#
#  Console commands:
#    start N [opts]     Start N headless instances (opts passed to tml-instances.sh)
#    stop [i]           Stop instance i, or ALL instances
#    restart [i]        Restart instance i, or the whole fleet
#    status             Redraw the dashboard
#    watch              Live dashboard (refreshes every 2s, any key exits)
#    logs <i> [n]       Last n lines (default 40) of instance i's game log
#
#    train [args...]    Start parallel training (auto num-envs/base-port,
#                       extra args passed to train_parallel.py, e.g. --new)
#    train stop         Stop training
#    trainlog [n]       Last n lines of the training log
#
#    boss <npc_id>      Set boss on ALL instances + apply boss_profiles.conf
#                       profile (arena platforms, time, world) if defined
#    arena <X> <Y>      Set arena center (world pixel coords) fleet-wide
#    time <v>           day|noon|night|midnight|none — force time each episode
#    automl on|off      Auto-arm ML mode when agent connects (no '/bml on')
#    modset K=V         Set any BossMLMod config key on ALL instances
#                       (e.g. modset FrameSkip=1) — restart fleet to apply
#
#    tb                 Toggle TensorBoard (http://localhost:6006)
#    help               Show commands
#    quit               Exit console (fleet keeps running; 'stop' first to kill)
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER="$PROJECT/tml-instances.sh"
AGENT_DIR="$PROJECT/terraria_boss_agent"
VENV_PY="$PROJECT/.venv/bin/python"
ROOT="${TML_INSTANCES_ROOT:-$HOME/.local/share/TerrariaML/instances}"
FLEET_CONF="$ROOT/fleet.conf"
TRAIN_PID_F="$ROOT/train.pid"
TRAIN_LOG="$ROOT/train.log"
TB_PID_F="$ROOT/tensorboard.pid"

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; YELLOW='\033[33m'
RED='\033[31m'; CYAN='\033[36m'; ORANGE='\033[38;5;209m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✔${RESET} $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "  ${RED}✘ $*${RESET}"; }
info() { echo -e "  ${DIM}$*${RESET}"; }

[[ -x "$MANAGER" ]] || { err "tml-instances.sh not found next to fleet.sh"; exit 1; }

pid_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }
read_pid()  { [[ -f "$1" ]] && cat "$1" 2>/dev/null || echo ""; }
uptime_of() { ps -o etime= -p "$1" 2>/dev/null | tr -d ' ' || true; }
port_up()   { timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

declare -A BOSS_NAMES=(
    [4]="Eye of Cthulhu" [13]="Eater of Worlds" [35]="Skeletron" [50]="King Slime"
    [113]="Wall of Flesh" [125]="The Twins" [127]="Skeletron Prime" [134]="The Destroyer"
    [222]="Queen Bee" [245]="Golem" [262]="Plantera" [370]="Duke Fishron" [398]="Moon Lord"
    [636]="Empress of Light" [657]="Queen Slime" [668]="Deerclops"
)

# ── Fleet introspection ──────────────────────────────────────────────────────
instance_ids() { for d in "$ROOT"/i*/; do [[ -d "$d" ]] && basename "$d" | sed 's/^i0\{0,1\}//'; done; }

running_ports() {  # ports of instances whose socket is actually listening
    local i p
    for i in $(instance_ids); do
        p="$(cat "$ROOT/$(printf 'i%02d' "$i")/port" 2>/dev/null || true)"
        [[ -n "$p" ]] && port_up "$p" && echo "$p"
    done | sort -n
}

current_boss() {
    local f
    f="$(ls "$ROOT"/i*/save/ModConfigs/BossMLMod_BossMLConfig.json 2>/dev/null | head -1)"
    [[ -n "$f" ]] && python3 -c "import json;print(json.load(open('$f')).get('BossNpcType','?'))" 2>/dev/null || echo "?"
}

# ── Dashboard ────────────────────────────────────────────────────────────────
render() {
    clear
    echo -e "  ${BOLD}${ORANGE}⛏ Terraria RL — Fleet Console${RESET}   ${DIM}$(date '+%H:%M:%S')  ·  root: $ROOT${RESET}"
    echo -e "  ${DIM}────────────────────────────────────────────────────────────────${RESET}"

    # Instances
    echo -e "  ${BOLD}Instances${RESET}"
    local any=0 i dir gpid port state sock up
    for i in $(instance_ids); do
        any=1
        dir="$ROOT/$(printf 'i%02d' "$i")"
        gpid="$(read_pid "$dir/game.pid")"
        port="$(cat "$dir/port" 2>/dev/null || echo '?')"
        if pid_alive "$gpid"; then
            up="$(uptime_of "$gpid")"
            state="${GREEN}● running${RESET} ${DIM}(pid $gpid, up $up)${RESET}"
        else
            state="${RED}○ stopped${RESET}                       "
        fi
        if [[ "$port" != "?" ]] && port_up "$port"; then sock="${GREEN}listening${RESET}"; else sock="${DIM}closed${RESET}   "; fi
        echo -e "    i$i  $state  port ${CYAN}$port${RESET}: $sock"
    done
    [[ $any -eq 0 ]] && info "  none — 'start 4' to launch a fleet"

    # Boss
    local bid bname
    bid="$(current_boss)"
    bname="${BOSS_NAMES[$bid]:-}"
    [[ $any -eq 1 ]] && echo -e "    ${DIM}boss:${RESET} $bid ${DIM}${bname:+($bname)}${RESET}"

    # Trainer
    echo -e "\n  ${BOLD}Trainer${RESET}"
    local tpid; tpid="$(read_pid "$TRAIN_PID_F")"
    if pid_alive "$tpid"; then
        echo -e "    ${GREEN}● training${RESET} ${DIM}(pid $tpid, up $(uptime_of "$tpid"))${RESET}"
        local last
        last="$(grep -aE "win_rate|ep_reward|Connected|timesteps|fps" "$TRAIN_LOG" 2>/dev/null | tail -2 | sed 's/^/    /' | cut -c1-100)"
        [[ -n "$last" ]] && echo -e "${DIM}$last${RESET}"
    else
        echo -e "    ${RED}○ stopped${RESET}  ${DIM}— 'train' to launch against listening instances${RESET}"
    fi

    # TensorBoard
    local bpid; bpid="$(read_pid "$TB_PID_F")"
    if pid_alive "$bpid"; then
        echo -e "  ${BOLD}TensorBoard${RESET}  ${GREEN}● http://localhost:6006${RESET}"
    else
        echo -e "  ${BOLD}TensorBoard${RESET}  ${DIM}○ off — 'tb' to toggle${RESET}"
    fi

    echo -e "  ${DIM}────────────────────────────────────────────────────────────────${RESET}"
    echo -e "  ${DIM}start N · stop [i] · restart [i] · logs i · train [--new] · train stop"
    echo -e "  trainlog · boss <id> · arena X Y · time <day|night> · automl on|off"
    echo -e "  modset K=V · tb · watch · help · quit${RESET}"
}

# ── Commands ─────────────────────────────────────────────────────────────────
cmd_start() {
    local n="${1:-}"; shift || true
    [[ "$n" =~ ^[0-9]+$ ]] || { err "Usage: start <count> [tml-instances.sh opts]"; return; }
    mkdir -p "$ROOT"
    echo "COUNT=$n" > "$FLEET_CONF"; echo "ARGS=\"$*\"" >> "$FLEET_CONF"
    "$MANAGER" start -n "$n" "$@"
}

cmd_stop_fleet() { "$MANAGER" stop "${1:-}"; }

cmd_restart() {
    local i="${1:-}"
    if [[ -n "$i" ]]; then
        "$MANAGER" stop "$i"
    else
        "$MANAGER" stop
    fi
    local n=1 args=""
    [[ -f "$FLEET_CONF" ]] && source "$FLEET_CONF" && n="$COUNT" && args="${ARGS:-}"
    if [[ -n "$i" ]]; then
        # shellcheck disable=SC2086
        "$MANAGER" start -n "$n" $args   # start skips already-running ones
    else
        # shellcheck disable=SC2086
        "$MANAGER" start -n "$n" $args
    fi
}

cmd_logs() {
    local i="${1:-}" n="${2:-40}"
    [[ -n "$i" ]] || { err "Usage: logs <instance> [lines]"; return; }
    local f="$ROOT/$(printf 'i%02d' "$i")/log.txt"
    [[ -f "$f" ]] && tail -n "$n" "$f" || err "No log at $f"
}

cmd_train() {
    if [[ "${1:-}" == "stop" ]]; then
        local tpid; tpid="$(read_pid "$TRAIN_PID_F")"
        if pid_alive "$tpid"; then
            kill -INT "$tpid" 2>/dev/null; sleep 3
            pid_alive "$tpid" && { kill -TERM -- "-$tpid" 2>/dev/null || kill -TERM "$tpid" 2>/dev/null; }
            ok "Trainer stopped (model checkpointed on SIGINT)."
        else
            info "Trainer not running."
        fi
        rm -f "$TRAIN_PID_F"; return
    fi

    local tpid; tpid="$(read_pid "$TRAIN_PID_F")"
    pid_alive "$tpid" && { warn "Trainer already running (pid $tpid) — 'train stop' first."; return; }
    [[ -x "$VENV_PY" ]] || { err "venv python not found at $VENV_PY — run setup first."; return; }

    mapfile -t ports < <(running_ports)
    local n="${#ports[@]}"
    [[ $n -ge 1 ]] || { err "No instances with listening sockets — 'start N' first and wait for ready."; return; }

    info "Launching trainer: $n env(s), ports ${ports[0]}..${ports[$((n-1))]}"
    ( cd "$AGENT_DIR" && exec setsid "$VENV_PY" scripts/train_parallel.py \
        --num-envs "$n" --base-port "${ports[0]}" "$@" ) >> "$TRAIN_LOG" 2>&1 &
    echo $! > "$TRAIN_PID_F"
    ok "Trainer started (pid $(cat "$TRAIN_PID_F")) — 'trainlog' to inspect."
}

cmd_trainlog() { [[ -f "$TRAIN_LOG" ]] && tail -n "${1:-30}" "$TRAIN_LOG" || info "No training log yet."; }

cmd_modset() {
    local kv="${1:-}"
    [[ "$kv" == *"="* ]] || { err "Usage: modset Key=Value   (e.g. modset FrameSkip=1)"; return; }
    local key="${kv%%=*}" val="${kv#*=}"
    [[ "$key" == "Port" ]] && { err "Port is managed per-instance — not settable fleet-wide."; return; }
    local count=0 f
    for f in "$ROOT"/i*/save/ModConfigs/BossMLMod_BossMLConfig.json; do
        [[ -f "$f" ]] || continue
        python3 - "$f" "$key" "$val" <<'PY'
import json, sys
path, key, raw = sys.argv[1], sys.argv[2], sys.argv[3]
try: val = json.loads(raw)          # numbers/bools pass through, else string
except Exception: val = raw
with open(path) as fh: cfg = json.load(fh)
cfg[key] = val
with open(path, "w") as fh: json.dump(cfg, fh, indent=2)
PY
        count=$((count+1))
    done
    [[ $count -gt 0 ]] && { ok "Set $key=$val on $count instance(s). 'restart' fleet to apply."; return 0; }
    err "No instance configs found — 'start N' first."; return 1
}

cmd_boss() {
    local id="${1:-}"
    [[ "$id" =~ ^[0-9]+$ ]] || { err "Usage: boss <npc_id>  (e.g. 4=EoC, 50=King Slime)"; return; }
    cmd_modset "BossNpcType=$id" || return 1
    local name="${BOSS_NAMES[$id]:-unknown}"
    info "Boss → $id ($name)"
    apply_boss_profile "$id"
}

# ── Boss profiles: per-stage arena worlds / platforms / time ────────────────
apply_boss_profile() {
    local id="$1" profile="$PROJECT/boss_profiles.conf"
    [[ -f "$profile" ]] || return 0
    local line
    line="$(grep -E "^${id}[[:space:]]" "$profile" | head -1 | sed 's/#.*//')"
    [[ -n "$line" ]] || return 0
    info "Applying profile: $line"
    local tok key val need_restart=0
    for tok in $line; do
        [[ "$tok" == *"="* ]] || continue
        key="${tok%%=*}"; val="${tok#*=}"
        case "$key" in
            time)   cmd_modset "ForceTimeOfDay=$val" >/dev/null && info "  time → $val" ;;
            world)  set_fleet_arg "--world" "$val";  need_restart=1 ;;
            player) set_fleet_arg "--player" "$val"; need_restart=1 ;;
            *)      cmd_modset "$key=$val" >/dev/null && info "  $key → $val" ;;
        esac
    done
    warn "Profile applied to instance configs — 'restart' fleet to take effect."
    [[ $need_restart -eq 1 ]] && warn "World/player changed: restart also re-clones if used with 'start --fresh'."
}

set_fleet_arg() {  # set_fleet_arg --world MyArena  (persist into fleet.conf ARGS)
    local flag="$1" val="$2" n=1 args=""
    [[ -f "$FLEET_CONF" ]] && source "$FLEET_CONF" && n="$COUNT" && args="${ARGS:-}"
    # strip existing occurrence of the flag and its value, then append
    args="$(echo " $args " | sed -E "s/ $flag [^ ]+ / /g" | xargs || true)"
    args="$args $flag $val"
    echo "COUNT=$n" > "$FLEET_CONF"; echo "ARGS=\"$args\"" >> "$FLEET_CONF"
    info "  fleet ${flag#--} → $val (saved for next start/restart)"
}

cmd_arena() {
    local x="${1:-}" y="${2:-}"
    [[ "$x" =~ ^-?[0-9]+$ && "$y" =~ ^-?[0-9]+$ ]] \
        || { err "Usage: arena <X> <Y>   (world PIXEL coords; tiles × 16)"; return; }
    cmd_modset "ArenaCenterX=$x" >/dev/null && cmd_modset "ArenaCenterY=$y" >/dev/null \
        && ok "Arena center → ($x, $y) on all instances. 'restart' to apply."
}

cmd_automl() {
    case "${1:-}" in
        on)  cmd_modset "AutoEnableML=true"  >/dev/null && ok "AutoEnableML ON — instances arm ML mode when the agent connects (patched mod required)." ;;
        off) cmd_modset "AutoEnableML=false" >/dev/null && ok "AutoEnableML OFF — '/bml on' needed manually." ;;
        *)   err "Usage: automl on|off" ;;
    esac
}

cmd_time() {
    case "${1:-}" in
        day|noon|night|midnight|none)
            cmd_modset "ForceTimeOfDay=$1" >/dev/null \
                && ok "ForceTimeOfDay → $1 (applied each episode reset; patched mod required)." ;;
        *)  err "Usage: time day|noon|night|midnight|none" ;;
    esac
}

cmd_tb() {
    local bpid; bpid="$(read_pid "$TB_PID_F")"
    if pid_alive "$bpid"; then
        kill "$bpid" 2>/dev/null; rm -f "$TB_PID_F"; ok "TensorBoard stopped."
    else
        local tb="$PROJECT/.venv/bin/tensorboard"
        [[ -x "$tb" ]] || { err "tensorboard not found in venv."; return; }
        ( cd "$AGENT_DIR" && exec "$tb" --logdir logs --port 6006 --bind_all ) >> "$ROOT/tensorboard.log" 2>&1 &
        echo $! > "$TB_PID_F"
        ok "TensorBoard → http://localhost:6006"
    fi
}

cmd_watch() {
    while true; do
        render
        echo -e "\n  ${DIM}watching — press any key to return to prompt${RESET}"
        read -r -t 2 -n 1 && break
    done
}

dispatch() {
    local cmd="${1:-}"; shift || true
    cmd="${cmd#/}"   # accept /start and start alike
    case "$cmd" in
        start)    cmd_start "$@";      render_after=1 ;;
        stop)     cmd_stop_fleet "$@"; render_after=1 ;;
        restart)  cmd_restart "$@";    render_after=1 ;;
        status|s) render ;;
        watch|w)  cmd_watch; render ;;
        logs|l)   cmd_logs "$@" ;;
        train|t)  cmd_train "$@" ;;
        trainlog|tl) cmd_trainlog "$@" ;;
        boss|b)   cmd_boss "$@" ;;
        modset|m) cmd_modset "$@" ;;
        arena)    cmd_arena "$@" ;;
        automl)   cmd_automl "$@" ;;
        time)     cmd_time "$@" ;;
        tb)       cmd_tb ;;
        help|h)   sed -n '2,34p' "$0" | sed 's/^#[ ]\{0,2\}//' ;;
        quit|q|exit) echo; info "Fleet keeps running. 'fleet.sh stop' to shut it down later."; exit 0 ;;
        "")       ;;
        *)        err "Unknown command: $cmd  ('help' for commands)" ;;
    esac
}

# ── Entry ────────────────────────────────────────────────────────────────────
mkdir -p "$ROOT"

if [[ $# -gt 0 ]]; then     # one-shot mode: fleet.sh status / fleet.sh stop / ...
    dispatch "$@"
    exit 0
fi

render
while true; do
    render_after=0
    echo
    if ! read -r -e -p "$(echo -e "  ${BOLD}${ORANGE}fleet>${RESET} ")" line; then
        echo; exit 0
    fi
    [[ -n "$line" ]] && history -s "$line" 2>/dev/null
    # shellcheck disable=SC2086
    dispatch $line
    [[ $render_after -eq 1 ]] && { sleep 0.5; render; }
done
