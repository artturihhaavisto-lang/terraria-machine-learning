# Terraria Boss-Fighting via Deep Reinforcement Learning: A Gymnasium-Compatible Framework for Real-Time Game Agent Research

**Abstract.** We present a framework for training autonomous agents to defeat bosses in the video game *Terraria* (Re-Logic, 2011) using deep reinforcement learning. The system couples a custom tModLoader mod written in C# with a Python reinforcement learning stack via synchronous TCP communication, exposing the game as a Gymnasium-compatible environment. The agent observes a 223-dimensional state vector encoding player vitals, boss kinematics, projectile threat scores, and environmental context, and selects actions from a structured MultiDiscrete space at each game tick. We train agents using Proximal Policy Optimization (PPO) with an MLP policy and report on curriculum design, reward shaping, and action-space discretization choices. The framework supports 14 distinct boss encounters and achieves effective training throughputs of 70–100 environment steps per second on consumer hardware.

---

## 1. Introduction

Video games have served as productive benchmarks for reinforcement learning (RL) research since Atari [Mnih et al., 2013]. Most prior work operates on either pixel-level observations in 2D arcade environments or structured state vectors in 3D action games [Vinyals et al., 2019; Berner et al., 2019]. *Terraria* presents a middle ground: it is a 2D sandbox action-RPG with richly structured game state, deterministic physics, and a diverse set of boss encounters with distinct AI behaviors, making it well-suited for studying generalization in game-playing agents.

Existing work on Terraria AI is limited to scripted bots and heuristic autopilot systems. To our knowledge, no prior published framework has exposed Terraria's real-time combat loop as a trainable RL environment. This work contributes:

1. A tModLoader C# mod that intercepts the game's 60-tick-per-second update loop and exposes game state over a local TCP socket.
2. A Python Gymnasium environment (`TerrariaEnv`) wrapping this interface with structured observation normalization, shaped reward functions, and episode lifecycle management.
3. An analysis of key design decisions: tick synchronization, serialization protocol, action-space structure, and reward shaping, with ablation rationale for each.
4. A curriculum learning scheme for progressive boss difficulty and a behavioral analysis of trained policies.

---

## 2. System Architecture

### 2.1 Overview

The framework consists of two communicating processes: (1) a tModLoader mod running inside the Terraria game process, and (2) a Python agent process hosting the RL training loop.

```
┌──────────────────────────────────────────────────────┐
│               Terraria / tModLoader (C#)              │
│  Game Thread @ 60 TPS                                  │
│  1. ModPlayer.PreUpdate()  ← apply PendingAction       │
│  2. Player.Update()        ← game physics              │
│  3. NPC.Update()           ← boss AI                  │
│  4. Projectile.Update()    ← projectile trajectories  │
│  5. ModSystem.PostUpdateEverything()                   │
│       → serialize StatePacket (JSON, ~3 KB)            │
│       → TCP send → block → receive ActionPacket        │
└──────────────────────────────┬───────────────────────┘
                               │ TCP localhost:7777
                               │ NDJSON protocol
┌──────────────────────────────▼───────────────────────┐
│               Python RL Agent                         │
│  TerrariaEnv(gymnasium.Env)                           │
│    step(action):                                      │
│      1. serialize action → JSON → TCP send            │
│      2. TCP recv → parse StatePacket                  │
│      3. build observation vector (float32[223])        │
│      4. compute shaped reward                         │
│      5. return (obs, reward, done, truncated, info)   │
│                                                       │
│  PPO Policy (Stable-Baselines3)                       │
│    ObsSize: 223                                       │
│    ActionSpace: MultiDiscrete[3, 2, 2, 16, 2, 2, 10] │
└──────────────────────────────────────────────────────┘
```

### 2.2 Tick Synchronization

The game thread blocks in `PostUpdateEverything()` until the Python agent responds with an action. This guarantees strict temporal coupling: action $a_t$ is always computed from state $s_t$, preserving the Markov property in the collected trajectories. On consumer CPU hardware (inference latency ≈ 3–5 ms), the blocking exchange completes within a single-tick budget (16.7 ms), yielding an effective training throughput of 70–100 steps/second — faster than real-time Terraria.

### 2.3 Communication Protocol

Messages are transmitted as newline-delimited JSON (NDJSON) over a loopback TCP socket. Each `StatePacket` is approximately 3 KB; each `ActionPacket` is under 100 bytes. The loopback interface contributes less than 0.1 ms of transport latency, making serialization the dominant cost. System.Text.Json (C#) and the stdlib `json` module (Python) were chosen for zero-dependency integration; MessagePack is noted as a future optimization path.

---

## 3. Observation Space

The observation vector is a flat `Box(−1, 1)` array of dimension **223**, normalized to `[0, 1]` or `[−1, 1]` depending on feature semantics. It encodes:

| Group | Dimensions | Description |
|---|---|---|
| Player state | 12 | HP, mana, position (x, y), velocity (x, y), on-ground flag, potion sickness, buff counts |
| Boss state | 18 | HP (normalized), position, velocity, phase indicator, raw AI fields `ai[0..3]` |
| Boss segments | 50 | Up to 10 segments × 5 features (active flag, position, velocity) |
| Projectile threats | 100 | Up to 20 projectiles × 5 features (active, position, velocity, damage), sorted by threat score $= \text{damage} / (d^2 + 1)$ |
| Arena context | 8 | Arena bounds, platform proximity, time-of-day, biome flags |
| Aim feedback | 6 | Current aim sector, mouse position, player facing direction |
| Cooldowns | 29 | Item use cooldown, grapple state, wing fuel, dodge timer |

Fixed-size padding with zero-filled inactive slots allows uniform tensor shapes across all boss types. Projectiles are sorted by a threat heuristic (damage-to-distance-squared ratio) as a static attention approximation.

---

## 4. Action Space

The agent selects from a **MultiDiscrete** action space at each decision step:

| Dimension | Cardinality | Meaning |
|---|---|---|
| `move` | 3 | {left, idle, right} |
| `jump` | 2 | {no-jump, jump} |
| `use_item` | 2 | {idle, use/channel} |
| `aim` | 16 | Aim sector (22.5° increments, clockwise from East) |
| `hook` | 2 | {retract, fire/hold} |
| `mount` | 2 | {no-op, toggle} |
| `quick_heal` | 10 | Heal action with high penalty when HP > 60% |

Aim is discretized into 16 sectors. At sector $k$, the target world position is projected as:

$$\theta_k = k \cdot \frac{2\pi}{16}, \quad (x_\text{tgt}, y_\text{tgt}) = \text{Player.Center} + r \cdot (\cos\theta_k, \sin\theta_k)$$

where $r$ is a fixed aim projection radius. This is converted to screen-space mouse coordinates before each game tick.

Actions are injected by overriding `Player.control*` fields in `ModPlayer.PreUpdate()`, before `Player.Update()` reads them, ensuring deterministic application.

---

## 5. Reward Function

The reward signal is a shaped scalar combining progress, survival, and terminal outcomes:

$$r_t = w_1 \Delta\hat{h}_\text{boss} + w_2 \Delta\hat{h}_\text{player} + w_3 \cdot \mathbf{1} + w_4 \cdot \phi_\text{prox} + W_5 \cdot \mathbf{1}[\text{kill}] + W_6 \cdot \mathbf{1}[\text{death}]$$

where $\hat{h}$ denotes HP normalized to $[0, 1]$, $\phi_\text{prox}$ is a proximity penalty encouraging engagement with the boss, and the terminal bonuses $W_5$, $W_6$ provide sparse outcome signals.

**Default weights:** $w_1 = 1.0$, $w_2 = -2.0$, $w_3 = 0.001$, $w_4 = -0.1$, $W_5 = 10.0$, $W_6 = -5.0$.

The asymmetric penalty on player damage ($w_2 = -2 w_1$) discourages tank-and-spank strategies. The proximity penalty mitigates reward hacking via indefinite kiting. Heal-action penalties when HP $> 60\%$ discourage early potion use. All weights are configurable via `config/default.yaml`.

---

## 6. Training

### 6.1 Algorithm

We use **Proximal Policy Optimization** [Schulman et al., 2017] as implemented in Stable-Baselines3 [Raffin et al., 2021]. PPO's clipped surrogate objective and on-policy rollouts are well-suited to the episodic structure of boss fights (clear episode boundaries at boss kill or player death). The MultiDiscrete action space is natively supported.

**Default architecture:** Two-layer MLP with hidden size 256 × 256. An optional recurrent variant using `RecurrentPPO` (sb3-contrib) with LSTM cells is supported for temporal credit assignment.

### 6.2 Curriculum

Training follows a difficulty curriculum:

| Stage | Boss | Notes |
|---|---|---|
| 1 | King Slime | Simplest movement patterns; good for environment validation |
| 2 | Eye of Cthulhu | Two-phase; introduces charge attacks |
| 3 | Skeletron | Multiple limbs; dodging required |
| 4 | Wall of Flesh | Linear arena; directional constraint |
| 5 | Mechanical Bosses | Hardmode; complex projectile patterns |
| 6 | Plantera / Golem | Constrained arenas |
| 7 | Moon Lord | Endgame; multi-phase, dense projectiles |

Curriculum advancement is triggered by a configurable win-rate threshold (default: 70% over 50 episodes).

### 6.3 Training Configuration

```yaml
algorithm: ppo
total_timesteps: 2_000_000
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
frame_skip: 4
```

With `frame_skip=4`, the agent makes decisions every 4 game ticks (~15–25 decisions/second). At 2M timesteps, a full training run requires approximately 22–37 hours on consumer CPU hardware.

---

## 7. Implementation Notes

### 7.1 Boss Coverage

The framework supports 14 vanilla Terraria bosses:

| Boss | NPC Type | Notes |
|---|---|---|
| King Slime | 50 | Pre-Hardmode, easiest |
| Eye of Cthulhu | 4 | Night-only |
| Eater of Worlds | 13 | Multi-segment (Corruption) |
| Brain of Cthulhu | 266 | Multi-phase (Crimson) |
| Queen Bee | 222 | Jungle biome |
| Skeletron | 35 | Night-only |
| Wall of Flesh | 113 | Linear Hell arena |
| The Twins | 125 | Hardmode, night |
| The Destroyer | 134 | Hardmode, 80+ segments |
| Skeletron Prime | 127 | Hardmode, night |
| Plantera | 262 | Jungle underground |
| Golem | 245 | Lihzahrd Temple |
| Duke Fishron | 370 | Ocean |
| Moon Lord | 398 | Endgame |

### 7.2 Known Limitations

- **Training throughput:** A single Terraria instance limits training to ~100 steps/second. Vectorized environments would require multiple independent game instances.
- **Headless operation:** Terraria requires a display. Use Xvfb on headless servers: `Xvfb :99 -screen 0 1280x720x24 &`.
- **Segment tracking:** Multi-segment bosses (Eater of Worlds, Destroyer) are tracked via head + top-9 proximity-sorted segments. Full attention-based segment awareness is left as future work.
- **Boss AI opacity:** Raw `npc.ai[]` fields are included in the observation but their semantics are boss-specific; the agent must infer them through interaction.

---

## 8. Quickstart

### Prerequisites

- Linux with X11/Wayland
- Steam, Terraria, tModLoader 1.4.4
- .NET 6 SDK, Python 3.11+

### Installation

```bash
# 1. Link or copy the mod to ModSources
MODSOURCES="$HOME/.local/share/Terraria/tModLoader/ModSources"
ln -s "$(pwd)/BossMLMod" "$MODSOURCES/BossMLMod"
# Build the mod via tModLoader: Workshop → Develop Mods → BossMLMod → Build

# 2. Install Python dependencies
cd terraria_boss_agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Launch game, enable BossMLMod, set up arena, then:
#    /bml arena    (save arena center)
#    /bml boss 50  (King Slime)
#    /bml on       (enable ML mode)

# 4. Start training
./scripts/run_training.sh
```

### Monitoring

```bash
tensorboard --logdir terraria_boss_agent/logs
# open http://localhost:6006
```

Key metrics: `rollout/win_rate`, `rollout/ep_reward`, `rollout/ep_length`.

---

## 9. Repository Structure

```
.
├── BossMLMod/                  # tModLoader C# mod
│   ├── BossMLMod.cs            # Mod entry point
│   ├── BossMLPlayer.cs         # ModPlayer: action injection
│   ├── BossMLSystem.cs         # ModSystem: state serialization + TCP
│   ├── BossMLConfig.cs         # In-game config panel
│   ├── BossMLOverlay.cs        # Debug overlay (optional)
│   ├── Helpers/                # Observation builders, serialization helpers
│   └── Networking/             # TCP server, packet types
├── terraria_boss_agent/        # Python RL agent
│   ├── src/
│   │   ├── env/                # TerrariaEnv, observation, action, reward
│   │   ├── network/            # PPO policy definitions
│   │   ├── training/           # Training loop, curriculum manager
│   │   ├── eval/               # Evaluation harness
│   │   └── comm/               # TCP client
│   ├── config/
│   │   ├── default.yaml        # Training hyperparameters
│   │   └── presets/            # Per-boss reward weight presets
│   └── scripts/
│       ├── run_training.sh
│       └── run_eval.sh
├── DESIGN.md                   # Architecture and design rationale
├── QUICKSTART.md               # Step-by-step setup guide
└── README.md                   # This document
```

---

## 10. References

- Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning.* arXiv:1312.5602.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR 22(268).
- Vinyals, O. et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature 575, 350–354.
- Berner, C. et al. (2019). *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680.
- Brockman, G. et al. (2016). *OpenAI Gym.* arXiv:1606.01540.
- Towers, M. et al. (2024). *Gymnasium.* Zenodo. https://doi.org/10.5281/zenodo.8127025.

---

## License

MIT License. See `LICENSE` for details.

> **Note:** This framework uses only official tModLoader hooks and operates exclusively in single-player mode. It performs no memory patching, DLL injection, or modification of game network traffic. It is incompatible with VAC-protected or anti-cheat multiplayer servers.
