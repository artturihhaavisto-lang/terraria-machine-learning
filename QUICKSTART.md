# Quickstart: Zero to Training Run

## Prerequisites

- Kubuntu 25.10 (or any Linux with X11/Wayland)
- Steam with Terraria installed
- tModLoader 1.4.4 (install via Steam: Terraria → Properties → Betas → tModLoader)
- .NET 6 SDK (`sudo apt install dotnet-sdk-6.0`)
- Python 3.11+ (`sudo apt install python3.11 python3.11-venv`)

## Step 1: Install the tModLoader Mod

```bash
# Find your tModLoader ModSources directory
MODSOURCES="$HOME/.local/share/Terraria/tModLoader/ModSources"
mkdir -p "$MODSOURCES"

# Symlink or copy the mod
ln -s "$(pwd)/BossMLMod" "$MODSOURCES/BossMLMod"

# Build the mod via tModLoader:
# 1. Launch tModLoader (via Steam or directly)
# 2. Go to Workshop → Develop Mods
# 3. Find BossMLMod → click "Build"
# 4. Enable the mod in Mods menu → Reload
```

Alternatively, build from command line if `tModLoader` CLI is available:
```bash
cd "$MODSOURCES/BossMLMod"
dotnet build  # For IDE code completion only — actual mod build happens in-game
```

## Step 2: Set Up the Python Environment

```bash
cd terraria_boss_agent

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Verify
python -c "from src.env.terraria_env import TerrariaEnv; print('OK')"
```

## Step 3: Prepare the Game

1. **Launch tModLoader** with BossMLMod enabled
2. **Create or load a character and world** (any world size, any character)
3. **Build a simple arena**: flat platform ~200 tiles wide, with campfire and heart lantern
4. **Equip your character**: appropriate weapons/armor for the boss you want to train against
5. **In-game chat commands**:
   ```
   /bml arena          ← saves current position as arena center
   /bml boss 50        ← set boss to King Slime (NPC type 50)
   /bml time noon      ← (or /bml time night for night bosses like EoC)
   /bml status         ← verify everything is configured
   /bml on             ← ENABLE ML MODE (agent takes control)
   ```

## Step 4: Start Training

```bash
cd terraria_boss_agent
source .venv/bin/activate

# Launch training (connects to game on localhost:7777)
./scripts/run_training.sh
```

You should see:
```
Connected to Terraria at 127.0.0.1:7777
Using PPO (MLP policy)
Starting training for 2000000 timesteps
```

The game will visibly play itself — the character will move, jump, and attack autonomously.

## Step 5: Monitor Training

```bash
# In a new terminal:
cd terraria_boss_agent
source .venv/bin/activate
tensorboard --logdir logs
# Open http://localhost:6006 in browser
```

Key metrics to watch:
- `rollout/win_rate` — fraction of episodes where the boss was killed
- `rollout/ep_reward` — total reward per episode (should trend upward)
- `rollout/ep_length` — episode length (should decrease as agent gets better)

## Step 6: Evaluate

```bash
./scripts/run_eval.sh --model logs/models/final_model.zip --episodes 20
```

## Common Boss NPC Type IDs

| Boss            | NPC Type ID | Notes                    |
|-----------------|-------------|--------------------------|
| King Slime      | 50          | Easiest, good for testing|
| Eye of Cthulhu  | 4           | Night only               |
| Eater of Worlds | 13          | Corruption, multi-segment|
| Brain of Cthulhu| 266         | Crimson                  |
| Queen Bee       | 222         | Jungle/hive              |
| Skeletron       | 35          | Night only               |
| Wall of Flesh   | 113         | Hell only, special arena |
| The Twins       | 125         | Hardmode, night          |
| The Destroyer   | 134         | Hardmode, night          |
| Skeletron Prime | 127         | Hardmode, night          |
| Plantera        | 262         | Jungle underground       |
| Golem           | 245         | Temple                   |
| Duke Fishron    | 370         | Ocean                    |
| Moon Lord       | 398         | Endgame                  |

## Troubleshooting

**"Connection refused"**: Game isn't running, mod isn't enabled, or port mismatch.
Check `/bml status` in-game. Default port is 7777.

**Agent doesn't move**: ML mode not active. Type `/bml on` in-game chat.

**Boss doesn't spawn**: Wrong conditions (night boss during day, biome requirements).
The framework doesn't override biome/time requirements — set them manually.

**Game freezes briefly each tick**: This is expected! The game blocks waiting for the
agent's response. If Python inference is fast (<16ms), it's barely noticeable.

**Headless training (no monitor)**: Use Xvfb:
```bash
sudo apt install xvfb
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 steam -applaunch 1281930  # tModLoader app ID
```

**Performance**: Training speed is bounded by game tick rate (~60-100 effective TPS).
With frame_skip=4 (default), the agent decides every 4 ticks, so effective
decision rate is ~15-25 decisions/sec. This means 2M timesteps ≈ 22-37 hours.
Reduce total_timesteps for faster experiments.
