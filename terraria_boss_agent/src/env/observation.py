"""
Observation processing: converts raw JSON state from the game into a
normalized fixed-size float32 numpy array.

The observation is fully boss-agnostic. Raw game state (positions, velocities,
HP fractions, AI floats, projectile data) is normalized and padded to a fixed
size. The agent learns what each field means through experience.

Feature counts are dynamic — optional features can be disabled via config
flags (include_mana, include_potion_sickness, include_item_onehot, include_time)
to reduce observation dimensionality for focused training.
"""

import math
import numpy as np
from gymnasium import spaces

# Fixed feature counts per category
BUFF_FEATURES_PER_SLOT = 2
BOSS_FEATURES = 11
SEGMENT_FEATURES = 4     # rel_x, rel_y, hp_frac, active
PROJECTILE_FEATURES = 5  # rel_x, rel_y, vel_x, vel_y, damage


def _player_feature_count(cfg: dict) -> int:
    """Compute number of player features based on config flags."""
    # Base: pos_x, pos_y, vel_x, vel_y, hp_frac, grounded, defense,
    #        flight_time, wings_available, direction, dash_cooldown = 11
    count = 11
    if cfg.get("include_mana", True):
        count += 1
    if cfg.get("include_potion_sickness", True):
        count += 1
    if cfg.get("include_grappled", True):
        count += 1
    return count


def _item_feature_count(cfg: dict) -> int:
    """Compute item feature count based on config flags."""
    if cfg.get("include_item_onehot", True):
        return 12  # 10 one-hot + damage + cooldown
    else:
        return 3   # selected_slot_norm + damage + cooldown


def _env_feature_count(cfg: dict) -> int:
    """Compute environment feature count based on config flags."""
    if cfg.get("include_time", True):
        return 3   # sin, cos, is_day
    else:
        return 0


def compute_obs_size(cfg: dict) -> int:
    """Compute observation vector size from config."""
    max_buffs = cfg.get("max_buffs", 22)
    max_seg = cfg.get("max_segments", 10)
    max_proj = cfg.get("max_projectiles", 20)
    return (_player_feature_count(cfg)
            + max_buffs * BUFF_FEATURES_PER_SLOT
            + _item_feature_count(cfg)
            + BOSS_FEATURES
            + max_seg * SEGMENT_FEATURES
            + max_proj * PROJECTILE_FEATURES
            + _env_feature_count(cfg))


def make_observation_space(cfg: dict | None = None) -> spaces.Box:
    """Create the flat Box observation space."""
    size = compute_obs_size(cfg or {})
    return spaces.Box(
        low=-1.0, high=1.0, shape=(size,), dtype=np.float32
    )


def process_state(state: dict, cfg: dict) -> np.ndarray:
    """
    Convert a raw state dict from the game into a normalized float32 array.

    All positions are relative to the player (egocentric frame).
    All values are normalized to roughly [-1, 1] or [0, 1].

    Args:
        state: Raw JSON state from the game.
        cfg: Observation config section from default.yaml.

    Returns:
        np.ndarray of shape (obs_size,), dtype=float32.
    """
    max_proj = cfg.get("max_projectiles", 20)
    max_seg = cfg.get("max_segments", 10)
    max_buffs = cfg.get("max_buffs", 22)
    obs_size = compute_obs_size(cfg)

    obs = np.zeros(obs_size, dtype=np.float32)
    idx = 0

    world_w = cfg.get("world_width", 67200.0)
    world_h = cfg.get("world_height", 19200.0)
    max_vel = cfg.get("max_velocity", 30.0)
    max_def = cfg.get("max_defense", 100.0)
    max_dmg = cfg.get("max_damage", 500.0)
    max_buff_dur = cfg.get("max_buff_duration", 216000.0)

    p = state.get("player", {})
    px = p.get("pos_x", 0.0)
    py = p.get("pos_y", 0.0)

    # --- Player features ---
    obs[idx] = px / world_w;               idx += 1
    obs[idx] = py / world_h;               idx += 1
    obs[idx] = _clamp(p.get("vel_x", 0.0) / max_vel); idx += 1
    obs[idx] = _clamp(p.get("vel_y", 0.0) / max_vel); idx += 1
    obs[idx] = _safe_frac(p.get("hp", 0), p.get("max_hp", 1)); idx += 1
    if cfg.get("include_mana", True):
        obs[idx] = _safe_frac(p.get("mana", 0), p.get("max_mana", 1)); idx += 1
    obs[idx] = 1.0 if p.get("grounded", False) else 0.0; idx += 1
    obs[idx] = min(p.get("defense", 0) / max_def, 1.0); idx += 1
    obs[idx] = _safe_frac(p.get("flight_time", 0), max(p.get("max_flight_time", 1), 1)); idx += 1
    if cfg.get("include_potion_sickness", True):
        obs[idx] = min(p.get("potion_sickness", 0) / 3600.0, 1.0); idx += 1
    obs[idx] = 1.0 if p.get("wings_available", False) else 0.0; idx += 1
    if cfg.get("include_grappled", True):
        obs[idx] = 1.0 if p.get("grappled", False) else 0.0; idx += 1
    obs[idx] = 1.0 if p.get("direction", 1) == 1 else 0.0; idx += 1
    obs[idx] = min(p.get("dash_cooldown", 0) / 30.0, 1.0); idx += 1  # ~30 tick dash delay

    # --- Buff slots — max_buffs x (type_norm, duration_norm) ---
    buffs = p.get("buffs", [])
    for i in range(max_buffs):
        if i < len(buffs):
            btype, bdur = buffs[i]
            obs[idx] = min(btype / 350.0, 1.0)
            obs[idx + 1] = min(bdur / max_buff_dur, 1.0)
        idx += 2

    # --- Item features ---
    sel = p.get("selected_item", 0)
    if cfg.get("include_item_onehot", True):
        for i in range(10):
            obs[idx] = 1.0 if i == sel else 0.0
            idx += 1
    else:
        obs[idx] = sel / 9.0; idx += 1  # normalized slot index
    obs[idx] = min(p.get("item_damage", 0) / max_dmg, 1.0); idx += 1
    obs[idx] = min(p.get("item_cooldown", 0) / max(p.get("item_use_time", 1), 1), 1.0); idx += 1

    # --- Primary boss features ---
    bosses = state.get("bosses", [])
    if bosses:
        b = bosses[0]
        bx = b.get("pos_x", px)
        by = b.get("pos_y", py)
        obs[idx] = _clamp((bx - px) / 2000.0);    idx += 1
        obs[idx] = _clamp((by - py) / 2000.0);    idx += 1
        obs[idx] = _clamp(b.get("vel_x", 0.0) / max_vel); idx += 1
        obs[idx] = _clamp(b.get("vel_y", 0.0) / max_vel); idx += 1
        obs[idx] = _safe_frac(b.get("hp", 0), b.get("max_hp", 1)); idx += 1
        obs[idx] = _clamp(b.get("ai0", 0.0) / 100.0); idx += 1
        obs[idx] = _clamp(b.get("ai1", 0.0) / 100.0); idx += 1
        obs[idx] = _clamp(b.get("ai2", 0.0) / 100.0); idx += 1
        obs[idx] = _clamp(b.get("ai3", 0.0) / 100.0); idx += 1
        obs[idx] = 1.0 if b.get("active", False) else 0.0; idx += 1
        dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
        obs[idx] = min(dist / 3000.0, 1.0); idx += 1
    else:
        idx += BOSS_FEATURES

    # --- Boss segments — up to max_seg x (rel_x, rel_y, hp_frac, active) ---
    segments = bosses[0].get("segments", []) if bosses else []
    for i in range(max_seg):
        if i < len(segments):
            s = segments[i]
            obs[idx] = _clamp((s.get("pos_x", px) - px) / 2000.0)
            obs[idx + 1] = _clamp((s.get("pos_y", py) - py) / 2000.0)
            obs[idx + 2] = _safe_frac(s.get("hp", 0), s.get("max_hp", 1))
            obs[idx + 3] = 1.0
        idx += SEGMENT_FEATURES

    # --- Hostile projectiles — up to max_proj x (rel_x, rel_y, vel_x, vel_y, damage) ---
    projs = state.get("projectiles", [])
    for i in range(max_proj):
        if i < len(projs):
            pr = projs[i]
            obs[idx] = _clamp((pr.get("pos_x", px) - px) / 1500.0)
            obs[idx + 1] = _clamp((pr.get("pos_y", py) - py) / 1500.0)
            obs[idx + 2] = _clamp(pr.get("vel_x", 0.0) / max_vel)
            obs[idx + 3] = _clamp(pr.get("vel_y", 0.0) / max_vel)
            obs[idx + 4] = min(pr.get("damage", 0) / max_dmg, 1.0)
        idx += PROJECTILE_FEATURES

    # --- Environment ---
    if cfg.get("include_time", True):
        env = state.get("environment", {})
        raw_time = env.get("time", 0.0)
        is_day = env.get("day_time", True)
        max_time = 54000.0 if is_day else 32400.0
        frac = raw_time / max(max_time, 1.0)
        obs[idx] = math.sin(2.0 * math.pi * frac); idx += 1
        obs[idx] = math.cos(2.0 * math.pi * frac); idx += 1
        obs[idx] = 1.0 if is_day else 0.0;         idx += 1

    assert idx == obs_size, f"Observation index {idx} != expected size {obs_size}"
    return obs


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _safe_frac(num: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, num / denom))
