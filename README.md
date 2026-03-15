# Terraria RL Agent

A real-time Reinforcement Learning agent that learns to defeat any boss in Terraria (including modded bosses) with zero human intervention.

## Architecture

```
┌──────────────────────────────┐     TCP Socket (localhost:7777)     ┌──────────────────────────────┐
│                              │  ── Observation JSON (per tick) ──► │                              │
│   tModLoader 1.4.4 C# Mod   │                                     │     Python ML Backend        │
│                              │  ◄── Action JSON (per tick) ──────  │                              │
│  - Game state extraction     │                                     │  - PPO agent (PyTorch)       │
│  - Player controller         │  ── Episode End Signal ──────────►  │  - Replay buffer             │
│  - Episode manager           │                                     │  - Training loop             │
│  - Boss auto-summoner        │  ◄── Ready Signal ───────────────   │  - Model save/load           │
│  - Arena builder             │                                     │  - Web dashboard             │
└──────────────────────────────┘                                     └──────────────────────────────┘
```

The system has two components communicating over a local TCP socket:
1. **tModLoader C# Mod** — hooks into the game, extracts observations, receives actions, controls the player, and manages the episode lifecycle.
2. **Python ML Backend** — runs PPO (Proximal Policy Optimization), receives streamed game state, returns actions, and trains between episodes.

## Prerequisites

- **Terraria** (Steam, v1.4.4+)
- **tModLoader** (Steam Workshop or standalone)
- **Python 3.10+**
- **CUDA-capable GPU** (recommended, CPU works but slower)

## Setup

### 1. Install the tModLoader Mod

```bash
# Copy the TerrariaRLAgent folder to your tModLoader mods source directory
# Typically: Documents/My Games/Terraria/tModLoader/ModSources/
cp -r TerrariaRLAgent/ ~/.local/share/Terraria/tModLoader/ModSources/

# Build the mod from within tModLoader's mod development menu
# Or use tModLoader CLI: dotnet build
```

### 2. Install Python Dependencies

```bash
cd terraria_rl
pip install -r requirements.txt
```

### 3. Configure

Edit `terraria_rl/configs/default.yaml` or create a custom config:

```yaml
environment:
  host: "localhost"
  port: 7777

reward:
  boss_killed: 100.0
  ideal_combat_range: 300.0

ppo:
  learning_rate: 0.0003
  rollout_length: 4096

training:
  device: "cuda"  # or "cpu"
```

### 4. In-Game Setup

1. Launch Terraria with tModLoader
2. Enable the "Terraria RL Agent" mod
3. Create or load a single-player world
4. **Equip your character** with the gear you want the agent to use (armor, weapons, accessories, potions)
5. Open the mod config (Settings → Mod Configuration → Terraria RL Agent):
   - Set the **Boss NPC Type ID** (see table below)
   - Set **Arena Center** coordinates (tile coords)
   - Adjust other settings as needed

## How to Select a Boss Target

Set the `BossNPCType` in the mod config to the NPC type ID:

| Boss | NPC Type ID |
|------|-------------|
| King Slime | 50 |
| Eye of Cthulhu | 4 |
| Eater of Worlds (head) | 13 |
| Brain of Cthulhu | 266 |
| Queen Bee | 222 |
| Skeletron | 35 |
| Deerclops | 668 |
| Wall of Flesh | 113 |
| Queen Slime | 657 |
| The Twins (Retinazer) | 125 |
| The Destroyer | 134 |
| Skeletron Prime | 127 |
| Plantera | 262 |
| Golem | 245 |
| Duke Fishron | 370 |
| Empress of Light | 636 |
| Lunatic Cultist | 439 |
| Moon Lord | 398 |

For **modded bosses**, find the NPC type ID in the mod's source code or use the Terraria wiki/mod documentation.

## How to Start Training

```bash
# Start training with default config (Eye of Cthulhu)
python terraria_rl/main.py --config terraria_rl/configs/default.yaml --mode train

# Train against King Slime
python terraria_rl/main.py --config terraria_rl/configs/king_slime.yaml --mode train

# Resume from checkpoint
python terraria_rl/main.py --config terraria_rl/configs/default.yaml --mode train --checkpoint checkpoints/checkpoint_500.pt
```

Then go in-game. The mod will wait for the Python client to connect, then automatically start episodes.

## Web Dashboard

The training dashboard starts automatically at `http://localhost:5555` when training begins.

Features:
- Real-time episode reward and win rate charts
- Live player/boss HP bars
- Spatial grid visualization (player, boss, projectiles, tiles)
- Policy/value loss and entropy graphs
- Training controls (pause, save, eval mode)
- Training log

## How to Watch the Agent Play (Inference Only)

```bash
python terraria_rl/main.py --config terraria_rl/configs/default.yaml --mode eval --checkpoint checkpoints/best_model.pt
```

This loads a trained model and runs inference without training. The agent plays at full speed.

## How to Switch to a Modded Boss

1. Find the modded boss's NPC type ID
2. Create a config file (or copy `configs/modded_boss_template.yaml`):
   ```yaml
   environment:
     host: "localhost"
     port: 7777
   reward:
     ideal_combat_range: 300.0  # Adjust for the boss
   ```
3. Set `BossNPCType` in the mod config to the modded boss's NPC type ID
4. Equip appropriate gear for the boss
5. Start training:
   ```bash
   python terraria_rl/main.py --config terraria_rl/configs/my_modded_boss.yaml --mode train
   ```

The observation schema is boss-agnostic — it reads generic NPC fields (position, HP, velocity, hitbox) that every boss has, vanilla or modded. No special handling is needed.

## Training Progression

| Episodes | Expected Behavior |
|----------|------------------|
| 0–100 | Random movement, dies instantly |
| 100–500 | Learns to move and survive a few seconds |
| 500–2,000 | Attacks boss, dodges some projectiles |
| 2,000–5,000 | Survives 30+ seconds, deals significant damage |
| 5,000–10,000 | Develops strategies, starts winning fights |
| 10,000+ | Wins consistently, optimizes time-to-kill |

Exact numbers depend on boss difficulty and player loadout. Leave training running overnight for best results.

## Arena System

The mod automatically generates a training arena with:
- Platforms spanning the configured width at regular vertical intervals
- Configurable platform spacing (default: every 10 tiles)
- Background walls removed for boss spawning

Configure in mod settings:
- `ArenaWidth` / `ArenaHeight` — size in tiles
- `PlatformInterval` — vertical spacing between platform rows
- `ArenaCenterX` / `ArenaCenterY` — center position

## Project Structure

```
terraria-machine-learning/
├── TerrariaRLAgent/              # tModLoader C# Mod
│   ├── TerrariaRLAgent.cs        # Mod entry point
│   ├── RLModPlayer.cs            # Player hooks
│   ├── RLModSystem.cs            # System hooks
│   ├── Networking/               # TCP socket server
│   ├── Observation/              # Game state extraction
│   ├── Control/                  # Player input control
│   ├── Episode/                  # Episode lifecycle
│   └── Config/                   # Mod configuration
├── terraria_rl/                  # Python ML Backend
│   ├── main.py                   # Training entry point
│   ├── agent/                    # PPO, policy network, buffers
│   ├── environment/              # Gym env, socket client, reward
│   ├── dashboard/                # Web UI dashboard
│   ├── utils/                    # Logging, checkpoints, normalization
│   └── configs/                  # YAML config files
└── README.md
```

## Key Design Decisions

- **No human data needed** — pure RL from scratch
- **Single-player only** — no multiplayer support
- **Normal game speed** — no time acceleration; leave running overnight
- **Boss-agnostic observations** — works with any vanilla or modded boss
- **Shaped rewards** — dense reward signal for faster learning
- **Frame stacking** — optional temporal context for complex boss patterns

## Hyperparameter Tuning Tips

- **Increase `rollout_length`** for bosses with long fights
- **Decrease `ideal_combat_range`** for melee builds
- **Increase `entropy_coef`** if the agent converges to one strategy too quickly
- **Decrease `learning_rate`** if training is unstable
- **Enable `frame_stack: 4`** for bosses with complex attack patterns (e.g., Moon Lord)

## License

MIT
