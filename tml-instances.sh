#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  tml-instances.sh — Manage parallel headless tModLoader instances for
#  RL training against the BossMLMod TCP interface.
#
#  Each instance gets:
#    • its own tModLoader save directory (cloned from your main one, so it
#      has the same player, world, mods and mod configs)
#    • its own TCP port for the BossMLMod socket server (base_port + i)
#    • its own virtual display (Xvfb :<display_base + i>) unless --render
#
#  Commands:
#    start [-n N] [opts]   Start N instances (default 2)
#    stop [i]              Stop all instances, or just instance i
#    status                Show instance state, ports, PIDs
#    logs <i> [-f]         Show (or follow) log of instance i
#    init [-n N] [opts]    Prepare instance save dirs without launching
#    clean                 Stop everything and delete instance save dirs
#
#  Start/init options:
#    -n, --count N         Number of instances                  [default: 2]
#    --base-port P         First BossMLMod TCP port             [default: 7777]
#    --display-base D      First Xvfb display number            [default: 90]
#    --render              Use the real display (no Xvfb) — instance 0 only
#    --tml-dir PATH        tModLoader install dir               [auto-detect]
#    --source-save PATH    tML save dir to clone from
#                          [default: ~/.local/share/Terraria/tModLoader]
#    --player NAME         Player (.plr name) to auto-load      [first found]
#    --world NAME          World (.wld name) to auto-load       [first found]
#    --fresh               Re-clone save dirs even if they exist
#
#  Examples:
#    ./tml-instances.sh start -n 4                # 4 headless instances, ports 7777-7780
#    ./tml-instances.sh start -n 8 --base-port 8000
#    ./tml-instances.sh status
#    ./tml-instances.sh logs 0 -f
#    ./tml-instances.sh stop
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

INSTANCES_ROOT="${TML_INSTANCES_ROOT:-$HOME/.local/share/TerrariaML/instances}"
SOURCE_SAVE="${TML_SOURCE_SAVE:-$HOME/.local/share/Terraria/tModLoader}"
TML_DIR="${TML_DIR:-}"
COUNT=2
BASE_PORT=7777
DISPLAY_BASE=90
RENDER=0
FRESH=0
PLAYER=""
WORLD=""

BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[32m'; YELLOW='\033[33m'
RED='\033[31m'; CYAN='\033[36m'; RESET='\033[0m'
info() { echo -e "    ${DIM}$*${RESET}"; }
ok()   { echo -e "    ${GREEN}✔${RESET} $*"; }
warn() { echo -e "    ${YELLOW}⚠${RESET} $*"; }
die()  { echo -e "    ${RED}✘ $*${RESET}" >&2; exit 1; }

usage() { sed -n '2,40p' "$0" | sed 's/^#[ ]\{0,2\}//'; exit 0; }

# ── Helpers ──────────────────────────────────────────────────────────────────
detect_tml_dir() {
    [[ -n "$TML_DIR" ]] && { echo "$TML_DIR"; return; }
    local candidates=(
        "$HOME/.local/share/Steam/steamapps/common/tModLoader"
        "$HOME/.steam/steam/steamapps/common/tModLoader"
        "$HOME/.steam/root/steamapps/common/tModLoader"
        "$HOME/tModLoader"
    )
    for c in "${candidates[@]}"; do
        [[ -f "$c/start-tModLoader.sh" || -f "$c/LaunchUtils/ScriptCaller.sh" ]] && { echo "$c"; return; }
    done
    echo ""
}

launcher_for() {
    local dir="$1"
    if   [[ -f "$dir/start-tModLoader.sh" ]];        then echo "$dir/start-tModLoader.sh"
    elif [[ -f "$dir/LaunchUtils/ScriptCaller.sh" ]]; then echo "$dir/LaunchUtils/ScriptCaller.sh"
    else echo ""
    fi
}

inst_dir()  { printf '%s/i%02d' "$INSTANCES_ROOT" "$1"; }
inst_port() {  # prefer the port recorded at prepare time (survives custom --base-port)
    local f; f="$(inst_dir "$1")/port"
    [[ -f "$f" ]] && { cat "$f"; return; }
    echo $(( BASE_PORT + $1 ))
}
inst_disp() { echo $(( DISPLAY_BASE + $1 )); }

port_listening() { timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; }

pid_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

read_pid() { [[ -f "$1" ]] && cat "$1" 2>/dev/null || echo ""; }

# ── Save-dir preparation ─────────────────────────────────────────────────────
prepare_instance() {
    local i="$1"
    local dir save port
    dir="$(inst_dir "$i")"; save="$dir/save"; port=$(( BASE_PORT + i ))
    mkdir -p "$dir"
    echo "$port" > "$dir/port"

    if [[ -d "$save" && $FRESH -eq 0 ]]; then
        info "instance $i: save dir exists — keeping (use --fresh to re-clone)."
    else
        rm -rf "$save"
        mkdir -p "$save"
        [[ -d "$SOURCE_SAVE" ]] || die "Source save dir not found: $SOURCE_SAVE
    Run tModLoader once normally (create player/world, build & enable BossMLMod), or pass --source-save."
        # Clone only what an instance needs: players, worlds, mods, configs
        for sub in Players Worlds Mods ModConfigs; do
            [[ -d "$SOURCE_SAVE/$sub" ]] && cp -a "$SOURCE_SAVE/$sub" "$save/$sub"
        done
        [[ -f "$SOURCE_SAVE/config.json" ]] && cp -a "$SOURCE_SAVE/config.json" "$save/config.json"
        ok "instance $i: cloned save dir from $SOURCE_SAVE"
    fi

    # Per-instance BossMLMod port
    python3 - "$save/ModConfigs/BossMLMod_BossMLConfig.json" "$port" <<'PY'
import json, os, sys
path, port = sys.argv[1], int(sys.argv[2])
cfg = {}
if os.path.exists(path):
    try:
        with open(path) as f: cfg = json.load(f)
    except Exception: cfg = {}
cfg["Port"] = port
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f: json.dump(cfg, f, indent=2)
PY

    # Make sure BossMLMod is enabled
    python3 - "$save/Mods/enabled.json" <<'PY'
import json, os, sys
path = sys.argv[1]
mods = []
if os.path.exists(path):
    try:
        with open(path) as f: mods = json.load(f)
    except Exception: mods = []
if "BossMLMod" not in mods: mods.append("BossMLMod")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f: json.dump(mods, f, indent=2)
PY
    ok "instance $i: port=$port  save=$save"
}

resolve_file() {  # resolve_file <dir> <ext> <preferred-name>
    local dir="$1" ext="$2" name="$3"
    if [[ -n "$name" ]]; then
        [[ -f "$dir/$name.$ext" ]] && { echo "$dir/$name.$ext"; return; }
        [[ -f "$dir/$name" ]] && { echo "$dir/$name"; return; }
        die "Not found: $dir/$name.$ext"
    fi
    local first
    first="$(find "$dir" -maxdepth 1 -name "*.$ext" 2>/dev/null | sort | head -1)"
    echo "$first"
}

# ── Commands ─────────────────────────────────────────────────────────────────
cmd_start() {
    local tml launcher
    tml="$(detect_tml_dir)"
    [[ -n "$tml" ]] || die "tModLoader install not found — pass --tml-dir /path/to/tModLoader"
    launcher="$(launcher_for "$tml")"
    [[ -n "$launcher" ]] || die "No start-tModLoader.sh in $tml"

    if [[ $RENDER -eq 0 ]] && ! command -v Xvfb &>/dev/null; then
        die "Xvfb not installed (sudo apt install xvfb), or use --render for a single visible instance."
    fi

    echo -e "${BOLD}Starting $COUNT instance(s)${RESET}  ${DIM}ports $(inst_port 0)-$(inst_port $((COUNT-1))), root: $INSTANCES_ROOT${RESET}"
    mkdir -p "$INSTANCES_ROOT"

    for (( i=0; i<COUNT; i++ )); do
        local dir save port disp gpid plr wld
        dir="$(inst_dir "$i")"; save="$dir/save"; port="$(inst_port "$i")"; disp="$(inst_disp "$i")"

        gpid="$(read_pid "$dir/game.pid")"
        if pid_alive "$gpid"; then
            warn "instance $i already running (pid $gpid) — skipping."
            continue
        fi

        prepare_instance "$i"
        plr="$(resolve_file "$save/Players" plr "$PLAYER")"
        wld="$(resolve_file "$save/Worlds"  wld "$WORLD")"
        [[ -n "$plr" && -n "$wld" ]] || die "instance $i: no player/world found in $save — create them in tModLoader first."

        # Virtual display
        local disp_arg=""
        if [[ $RENDER -eq 1 && $i -eq 0 ]]; then
            info "instance 0: rendering on real display ${DISPLAY:-:0}"
        else
            if ! [[ -e "/tmp/.X11-unix/X$disp" ]]; then
                Xvfb ":$disp" -screen 0 1280x720x24 -nolisten tcp &>/dev/null &
                echo $! > "$dir/xvfb.pid"
                sleep 0.3
            fi
            disp_arg=":$disp"
        fi

        # Launch the game in its own process group so stop() can kill the tree
        (
            cd "$tml"
            [[ -n "$disp_arg" ]] && export DISPLAY="$disp_arg"
            export SDL_AUDIODRIVER=dummy
            exec setsid bash "$launcher" \
                -tmlsavedirectory "$save" \
                -player "$plr" \
                -world "$wld" \
                >> "$dir/log.txt" 2>&1
        ) &
        echo $! > "$dir/game.pid"
        ok "instance $i: launched (pid $(cat "$dir/game.pid"), display ${disp_arg:-real}, port $port) — log: $dir/log.txt"
    done

    echo
    info "Waiting for BossMLMod sockets (mod binds its port once in-world)..."
    local deadline=$(( $(date +%s) + 180 )) up
    while true; do
        up=0
        for (( i=0; i<COUNT; i++ )); do port_listening "$(inst_port "$i")" && up=$((up+1)); done
        echo -ne "    ${DIM}$up/$COUNT sockets up...${RESET}\r"
        [[ $up -eq $COUNT ]] && { echo; ok "All $COUNT instances ready."; break; }
        [[ $(date +%s) -gt $deadline ]] && { echo; warn "$up/$COUNT ready after 180s — check './tml-instances.sh logs <i>'. If -player/-world autoload isn't supported by your tML version, join a world manually per instance."; break; }
        sleep 3
    done

    echo
    echo -e "${BOLD}Train against them:${RESET}"
    echo "    python terraria_boss_agent/scripts/train_parallel.py --num-envs $COUNT --base-port $BASE_PORT"
}

cmd_stop() {
    local only="${1:-}"
    local found=0
    for dir in "$INSTANCES_ROOT"/i*/; do
        [[ -d "$dir" ]] || continue
        local i; i="$(basename "$dir" | sed 's/^i0\{0,1\}//')"
        [[ -n "$only" && "$i" != "$only" ]] && continue
        found=1
        local gpid xpid
        gpid="$(read_pid "$dir/game.pid")"; xpid="$(read_pid "$dir/xvfb.pid")"
        if pid_alive "$gpid"; then
            kill -TERM -- "-$gpid" 2>/dev/null || kill -TERM "$gpid" 2>/dev/null || true
            sleep 1
            pid_alive "$gpid" && { kill -KILL -- "-$gpid" 2>/dev/null || kill -KILL "$gpid" 2>/dev/null || true; }
            ok "instance $i: game stopped."
        else
            info "instance $i: game not running."
        fi
        pid_alive "$xpid" && { kill "$xpid" 2>/dev/null || true; ok "instance $i: Xvfb stopped."; }
        rm -f "$dir/game.pid" "$dir/xvfb.pid"
    done
    [[ $found -eq 0 ]] && info "No instances found under $INSTANCES_ROOT."
}

cmd_status() {
    echo -e "${BOLD}Instances${RESET}  ${DIM}($INSTANCES_ROOT)${RESET}"
    local any=0
    for dir in "$INSTANCES_ROOT"/i*/; do
        [[ -d "$dir" ]] || continue
        any=1
        local i gpid state sock port
        i="$(basename "$dir" | sed 's/^i0\{0,1\}//')"
        port="$(inst_port "$i")"
        gpid="$(read_pid "$dir/game.pid")"
        if pid_alive "$gpid"; then state="${GREEN}running${RESET} (pid $gpid)"; else state="${RED}stopped${RESET}"; fi
        if port_listening "$port"; then sock="${GREEN}listening${RESET}"; else sock="${DIM}closed${RESET}"; fi
        echo -e "  i$i   game: $state   port $port: $sock   ${DIM}log: ${dir}log.txt${RESET}"
    done
    [[ $any -eq 0 ]] && info "None. Run: ./tml-instances.sh start -n 4"
}

cmd_logs() {
    local i="${1:-}"; shift || true
    [[ -n "$i" ]] || die "Usage: tml-instances.sh logs <i> [-f]"
    local dir; dir="$(inst_dir "$i")"
    [[ -f "$dir/log.txt" ]] || die "No log for instance $i ($dir/log.txt)"
    if [[ "${1:-}" == "-f" ]]; then tail -f "$dir/log.txt"; else tail -100 "$dir/log.txt"; fi
}

cmd_init() {
    mkdir -p "$INSTANCES_ROOT"
    for (( i=0; i<COUNT; i++ )); do prepare_instance "$i"; done
    ok "Prepared $COUNT instance save dirs (not launched)."
}

cmd_clean() {
    cmd_stop
    rm -rf "$INSTANCES_ROOT"
    ok "Removed $INSTANCES_ROOT"
}

# ── Parse ────────────────────────────────────────────────────────────────────
[[ $# -ge 1 ]] || usage
CMD="$1"; shift
POS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--count)     COUNT="$2"; shift 2 ;;
        --base-port)    BASE_PORT="$2"; shift 2 ;;
        --display-base) DISPLAY_BASE="$2"; shift 2 ;;
        --render)       RENDER=1; shift ;;
        --tml-dir)      TML_DIR="$2"; shift 2 ;;
        --source-save)  SOURCE_SAVE="$2"; shift 2 ;;
        --player)       PLAYER="$2"; shift 2 ;;
        --world)        WORLD="$2"; shift 2 ;;
        --fresh)        FRESH=1; shift ;;
        -h|--help)      usage ;;
        *)              POS+=("$1"); shift ;;
    esac
done

case "$CMD" in
    start)  cmd_start ;;
    stop)   cmd_stop "${POS[0]:-}" ;;
    status) cmd_status ;;
    logs)   cmd_logs "${POS[@]:-}" ;;
    init)   cmd_init ;;
    clean)  cmd_clean ;;
    -h|--help|help) usage ;;
    *)      die "Unknown command: $CMD (see --help)" ;;
esac
