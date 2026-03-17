# Terraria Boss-Fighting ML Framework — Design Document

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TERRARIA (Single-Player)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  tModLoader BossMLMod (C#)                                │  │
│  │                                                           │  │
│  │  Game Thread (60 TPS)                                     │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  1. ModPlayer.PreUpdate()                           │  │  │
│  │  │     └─ Apply PendingAction → Player.control*        │  │  │
│  │  │     └─ Set Main.mouseX/Y for aim                   │  │  │
│  │  │  2. Player.Update() ← game logic runs              │  │  │
│  │  │  3. NPC.Update() × all NPCs                        │  │  │
│  │  │  4. Projectile.Update() × all projectiles          │  │  │
│  │  │  5. ModSystem.PostUpdateEverything()                │  │  │
│  │  │     └─ Build StatePacket (JSON)                    │  │  │
│  │  │     └─ Send over TCP (BLOCKS game thread)          │  │  │
│  │  │     └─ Wait for ActionPacket (timeout: 100ms)      │  │  │
│  │  │     └─ Store as PendingAction                      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  TCPListener on localhost:7777                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ TCP localhost:7777
                              │ NDJSON (newline-delimited JSON)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  Python ML Agent                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  TerrariaEnv(gymnasium.Env)                               │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  step(action):                                      │  │  │
│  │  │    1. Serialize action → JSON → send over TCP       │  │  │
│  │  │    2. Receive state JSON from TCP                   │  │  │
│  │  │    3. Normalize observation → float32 array[223]   │  │  │
│  │  │    4. Compute shaped reward                         │  │  │
│  │  │    5. Return (obs, reward, done, truncated, info)   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  PPO Policy (SB3 + custom LSTM/MLP network)               │  │
│  │    ObsSize: 223  ActionSpace: MultiDiscrete[3,2,2,16,2,2,10]│ │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Per Tick

```
Tick N timeline (wall clock):
 0ms  — ModPlayer.PreUpdate(): apply PendingAction (from tick N-1) to Player.control*
 1ms  — Player.Update(): movement, item use, collision
 2ms  — NPC updates: boss AI, segment updates
 3ms  — Projectile updates: trajectory, collision
 4ms  — ModSystem.PostUpdateEverything():
          serialize StatePacket → JSON (~3KB)
          TCP write to Python (~0.1ms local loopback)
          TCP read block: wait for ActionPacket
10ms  — Python agent receives state, runs forward pass (~3ms on CPU)
13ms  — Python sends ActionPacket JSON (~0.05ms)
13ms  — Game receives action, stores as PendingAction
13ms  — Game thread unblocked, tick N complete
         Effective TPS: ~1000/13 ≈ 77 TPS (faster than real-time 60 TPS)
```

## Key Design Decisions

### 1. Game-as-Server vs Python-as-Server
**Decision:** Game is the TCP server, Python is the client.

**Rationale:** The game controls timing (ticks). It makes sense for the game to own
the listener and block until Python connects. Python can restart independently
without restarting the game. If Python crashes and reconnects, the mod just
re-accepts the connection.

**Alternative considered:** Python server. Rejected because it requires the game to
connect on startup, and reconnection logic is harder to manage mid-fight.

### 2. Tick Synchronization: Blocking vs Async
**Decision:** Blocking synchronous exchange in PostUpdateEverything().

**Rationale:** For RL training, action-observation pairs must be tightly coupled.
With blocking exchange:
- action_t is always computed from state_t (perfect Markov property)
- No stale observations in the replay buffer
- Simpler code: no lock queues, no race conditions

**Alternative considered:** Async with last-action fallback. Rejected for training
(produces incorrect (s,a) pairs) but noted as useful for inference/demo.

**Tick rate:** On CPU inference (~5ms), effective training TPS is ~70-100, faster
than real Terraria. On GPU, even faster. Game renders are still produced (at
reduced rate) which is useful for debugging but can be disabled via config.

### 3. Serialization: NDJSON vs MessagePack vs Raw Binary
**Decision:** Newline-delimited JSON (NDJSON) using System.Text.Json + Python json.

**Rationale:**
- System.Text.Json is built into .NET 6 — zero extra dependencies in the mod
- Python json module is built-in — zero extra Python deps
- Human-readable: critical for debugging during development
- ~3KB per tick, well within loopback bandwidth
- Latency matters more than bandwidth: loopback TCP adds <0.1ms regardless

**Alternative considered:** MessagePack — ~30% smaller, ~2x faster parse. Worth
switching to if profiling shows serialization is a bottleneck (unlikely at 60Hz).

**Protocol:** Each message is a single JSON object followed by `\n`. Reader buffers
until `\n`, then parses. This is simple, robust, and handles partial reads.

### 4. Input Injection: ModPlayer.PreUpdate()
**Decision:** Override Player.control* fields in ModPlayer.PreUpdate().

**Rationale:** In tModLoader 1.4.4, player inputs (keyboard/gamepad) are processed
*before* Player.Update() is called. By the time PreUpdate() runs, control fields
are already set from hardware. Overriding them in PreUpdate() is correct because
Player.Update() reads them fresh each tick.

**Critical:** We must override ALL control fields, not just the ones we want active.
Otherwise, if the human pressed Jump the previous tick, that control stays set.

**Edge cases handled:**
- **Grappling hook mid-swing:** controlHook=false keeps the hook retracted. Setting
  controlHook=true while a hook is mid-flight retracts it. Agent learns to keep
  hook active by holding controlHook=1.
- **Channel items (Last Prism, Nimbus Rod):** controlUseItem=true while item is
  animating keeps channel active. Agent can hold use_item=1 for sustained beams.
- **Mount toggle:** controlMount=true triggers mount mount/dismount. Agent should
  rarely use this; we include it but penalize mount use during boss fights.
- **useTime cooldown:** Items have a useTime counter. controlUseItem=true while
  cooldown > 0 is a no-op (Terraria ignores it). Agent learns naturally that
  spamming use_item has no extra effect.
- **Wings/Rocket Boots:** controlUp while airborne uses wings. No special handling
  needed; agent learns to hold up during flight.

### 5. Aim Control: Discretized Sectors
**Decision:** 16 discrete aim sectors (22.5° each), 0=East, going clockwise.

**Rationale:**
- Continuous aim angle would require Hybrid or Continuous action space
- PPO with MultiDiscrete is simpler and converges faster for discrete control
- 16 sectors gives sufficient granularity (22.5° error at 1000px = 390px off-target,
  acceptable for AoE weapons and reasonable for targeted weapons)
- Implementation: compute angle = sector × 22.5°, project to screen coords

**Conversion formula:**
```
aimAngleRad = sector * (2π / 16)
targetWorldX = Player.Center.X + cos(aimAngle) * AIM_DISTANCE
targetWorldY = Player.Center.Y + sin(aimAngle) * AIM_DISTANCE
Main.mouseX = (int)(targetWorldX - Main.screenPosition.X)
Main.mouseY = (int)(targetWorldY - Main.screenPosition.Y)
// Player.direction: sector 4-11 (pointing left hemisphere) → direction = -1, else 1
Player.direction = (sector >= 4 && sector <= 11) ? -1 : 1
Player.itemRotation = (float)Math.Atan2(sin(aimAngle)*direction, cos(aimAngle)*direction)
```

### 6. Observation Space Design
**Decision:** Flat Box[223], normalized to [−1, 1] or [0, 1].

**Rationale:**
- Flat Box works well with MLP policies in SB3
- Fixed size: variable-length boss segments and projectiles are padded to fixed slots
- Sorted by relevance: projectiles sorted by threat score (damage/distance²)
- Alternative considered: Dict space — more interpretable but requires custom policy
  The flat Box is easier to use with SB3's built-in MLP and also with custom networks

### 7. Variable-Length Inputs (Segments, Projectiles)
**Decision:** Fixed-size padded arrays, sorted by threat/proximity.

**Segments:** Up to 10 segments. Extra slots filled with zeros (active=0 as sentinel).
**Projectiles:** Up to 20 projectiles. Sorted by threat = damage / (distance² + 1).
Extra slots filled with zeros. This is a static attention approximation.

**Alternative considered:** Attention mechanism (Transformer encoder over projectiles).
Worth implementing if agent struggles with projectile dodging. Start simple first.

### 8. Reward Function
Primary objectives (boss-agnostic):
```
r_t = w1 * Δboss_hp_norm     (damage dealt, positive)
    + w2 * Δplayer_hp_norm    (damage taken, negative because Δ is negative)
    + w3 * survival_bonus     (tiny positive per step)
    + w4 * proximity_reward   (moderate penalty if too far from boss)
    + W5 * boss_kill_bonus    (large terminal, +1)
    + W6 * death_penalty      (large terminal, -1)
```

Default weights: w1=1.0, w2=-2.0, w3=0.001, w4=-0.1, W5=10.0, W6=-5.0

**Reward hacking concerns:**
- Agent might kite forever (survival > kill) → proximity penalty mitigates this
- Agent might spam heal (if heal is an action) → restrict quick_heal to a sub-action
  with a high penalty if used when hp > 60% (informed by potion_sickness obs)
- Agent might die-reset-spawn loop to avoid long fights → episode length limit

### 9. PPO vs SAC
**Decision:** PPO (Proximal Policy Optimization) via Stable-Baselines3.

**Rationale:**
- Action space is MultiDiscrete → PPO handles this natively in SB3
- SAC requires continuous action space (or significant modification)
- PPO is on-policy: stable training, predictable behavior
- Environment has clear episode boundaries (boss kill or player death)
- PPO's clipped surrogate objective prevents destructive policy updates

**Architecture:** Two-layer MLP (256×256) by default. Optional LSTM for temporal
memory (use CnnLstmPolicy equivalent → RecurrentPPO from sb3-contrib).
Frame stacking (4 frames) as simpler alternative to LSTM.

## Known Limitations

1. **Training speed:** Single Terraria instance limits to ~100 environment steps/sec.
   Parallel instances would require multiple Terraria processes — difficult on Linux.
   Workaround: Keep episode resets fast (instant via mod command), minimize game
   load time with a dedicated training save file.

2. **Headless mode:** Terraria requires a display. Use Xvfb virtual display for
   headless training on a server: `Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 ...`

3. **Boss diversity:** AI fields (npc.ai[]) are boss-specific. The observation
   includes raw ai[] values, but their meaning varies by boss. Agent must learn
   per-boss AI semantics through experience.

4. **Multi-segment bosses:** Eater of Worlds, Destroyer have 100+ segments.
   We track the head + top-9 segments by proximity to player. Full segment
   awareness would require attention mechanism (future work).

5. **Input injection vs anti-cheat:** This mod only uses tModLoader hooks —
   no memory patching, no DLL injection. It works only in single-player and
   local mods; it cannot be used on VAC-protected or anti-cheat servers.

## Future Improvements

- Switch to MessagePack for 2x serialization speed
- Attention-based policy for projectile handling
- Multi-environment vectorized training with Xvfb instances
- Curriculum progression: King Slime → EoC → Skeletron → WoF → hardmode
- Imitation learning warm-start: record human play, pre-train via behavioral cloning
- Self-play arena for PvP training
