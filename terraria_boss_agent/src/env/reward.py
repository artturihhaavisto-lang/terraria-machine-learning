"""
Boss-agnostic reward shaping.

The reward function uses only generic game state signals that apply to any boss:
  - Damage dealt (boss HP decreased)
  - Damage taken (player HP decreased)
  - Survival (alive per step)
  - Proximity (stay within fighting range)
  - Terminal: boss kill bonus, death penalty, despawn penalty

No boss-specific knowledge is encoded. The agent learns optimal behavior
for each boss purely from these universal signals.

Reward formula:
  r_t = w1 * (boss_hp_delta / boss_max_hp)      # Positive when dealing damage
      - w2 * (player_hp_delta / player_max_hp)   # Positive when taking damage (penalty)
      + w3                                        # Survival bonus per step
      - w4 * max(0, dist - threshold)             # Distance penalty (linear)
      - w8 * max(0, dist - threshold)^2           # Distance penalty (quadratic, harsh)
      + W5 * boss_killed                          # Terminal bonus
      - W6 * player_died                          # Terminal penalty
      - w7 * wasteful_heal                        # Penalty for healing above 60% HP
      - W9 * boss_despawned                       # Penalty for boss despawning (kiting)

Hot reload: the reward section of the config YAML is re-read at each episode
reset, so you can tweak weights while training is running.
"""

import logging
import math
import os
import yaml

logger = logging.getLogger(__name__)

# Terraria: 1 block = 16 pixels
PIXELS_PER_BLOCK = 16

# Weight keys and their defaults
_WEIGHT_DEFAULTS = {
    "boss_damage_weight": 5.0,
    "player_damage_weight": 200.0,
    "survival_bonus": 0.0,
    "distance_penalty_weight": 0.1,
    "distance_sq_penalty_weight": 0.0,
    "distance_threshold": 240.0,
    "boss_kill_bonus": 100.0,
    "death_penalty": 100.0,
    "wasteful_heal_penalty": 0.0,
    "despawn_penalty": 100.0,
    "proximity_radius_blocks": 15,
    "proximity_bonus": 0.0,
    "dps_bonus_weight": 1.0,
    "idle_penalty_per_sec": 1.0,
    "flawless_kill_bonus": 0.0,
    "dodge_streak_bonus": 0.0,
    "dodge_streak_radius_blocks": 15,
    "nohit_bonus": 0.0,
    "hit_compounding_penalty": 5.0,
    "nohit_kill_bonus": 10000.0,
}


class RewardCalculator:
    """Computes shaped reward from raw reward components.

    Supports hot-reloading: if config_path is set, reward weights are
    re-read from the YAML file at each episode reset.
    """

    def __init__(self, cfg: dict, config_path: str | None = None):
        self._config_path = config_path
        # Initialize mtime to current file time so we only reload on *new* changes
        if config_path and os.path.exists(config_path):
            self._last_mtime = os.path.getmtime(config_path)
        else:
            self._last_mtime: float = 0.0
        self.pending_message: str | None = None
        self._total_boss_dmg_frac: float = 0.0
        self._step: int = 0
        self._idle_steps: int = 0
        self._hits_taken: int = 0
        self._dodge_streak: int = 0
        self._player_died_this_episode: bool = False
        self._apply_weights(cfg)

    def _apply_weights(self, cfg: dict) -> None:
        self.w_boss_dmg = cfg.get("boss_damage_weight", 1.0)
        self.w_player_dmg = cfg.get("player_damage_weight", 2.0)
        self.w_survival = cfg.get("survival_bonus", 0.001)
        self.w_dist = cfg.get("distance_penalty_weight", 0.0001)
        self.w_dist_sq = cfg.get("distance_sq_penalty_weight", 0.0)
        self.dist_thresh = cfg.get("distance_threshold", 800.0)
        self.w_kill = cfg.get("boss_kill_bonus", 10.0)
        self.w_death = cfg.get("death_penalty", 5.0)
        self.w_wasteful_heal = cfg.get("wasteful_heal_penalty", 0.5)
        self.w_despawn = cfg.get("despawn_penalty", 10.0)
        self.proximity_radius = cfg.get("proximity_radius_blocks", 25) * PIXELS_PER_BLOCK
        self.w_proximity = cfg.get("proximity_bonus", 0.002)
        self.w_dps_bonus = cfg.get("dps_bonus_weight", 1.0)
        self.w_idle_penalty = cfg.get("idle_penalty_per_sec", 0.005)
        self.w_flawless = cfg.get("flawless_kill_bonus", 5.0)
        self.w_dodge_streak = cfg.get("dodge_streak_bonus", 0.001)
        self.dodge_streak_radius = cfg.get("dodge_streak_radius_blocks", 20) * PIXELS_PER_BLOCK
        self.w_nohit = cfg.get("nohit_bonus", 0.0)
        self.w_hit_compound = cfg.get("hit_compounding_penalty", 0.0)
        self.w_nohit_kill = cfg.get("nohit_kill_bonus", 0.0)

    def _try_hot_reload(self) -> None:
        """Re-read reward weights from YAML if the file has changed."""
        if not self._config_path:
            return
        try:
            mtime = os.path.getmtime(self._config_path)
            if mtime <= self._last_mtime:
                return
            self._last_mtime = mtime
            with open(self._config_path, "r") as f:
                full_cfg = yaml.safe_load(f) or {}
            new_reward = full_cfg.get("reward", {})
            self._apply_weights(new_reward)
            logger.info("[HOT RELOAD] Reward weights reloaded")
            self.pending_message = "Config reloaded"
        except Exception as e:
            logger.warning(f"[HOT RELOAD] Failed to reload config: {e}")

    def reset(self) -> None:
        """Reset per-episode tracking. Also checks for config hot-reload."""
        self._total_boss_dmg_frac = 0.0
        self._step = 0
        self._idle_steps = 0
        self._hits_taken = 0
        self._dodge_streak = 0
        self._player_died_this_episode = False
        self._try_hot_reload()

    def compute(self, state: dict, action: dict) -> float:
        """
        Compute shaped reward from a state dict.

        Args:
            state: Raw state from game.
            action: Action dict that was taken (to check for wasteful heals).

        Returns:
            Scalar reward.
        """
        rc = state.get("reward_components", {})
        p = state.get("player", {})
        bosses = state.get("bosses", [])

        self._step += 1
        reward = 0.0

        # --- Damage dealt to boss (positive reward, with DPS bonus) ---
        boss_max_hp = 1.0
        if bosses:
            b = bosses[0]
            boss_max_hp = max(b.get("max_hp", 1), 1)

        # Check if boss is in an immunity phase
        boss_immune = False
        if bosses:
            boss_immune = bosses[0].get("immune", False)

        boss_hp_delta = rc.get("boss_hp_delta", 0.0)
        dealt_damage = boss_max_hp > 0 and boss_hp_delta < 0

        if dealt_damage:
            # boss_hp_delta is negative when boss takes damage
            dmg_frac = -boss_hp_delta / boss_max_hp

            if self.w_dps_bonus > 0:
                self._total_boss_dmg_frac += dmg_frac
                seconds = max(self._step / 15.0, 0.1)
                dps_rate = self._total_boss_dmg_frac / seconds
                dps_multiplier = 1.0 + self.w_dps_bonus * dps_rate
            else:
                dps_multiplier = 1.0

            reward += self.w_boss_dmg * dmg_frac * dps_multiplier
            self._idle_steps = 0
        elif self.w_idle_penalty > 0 and not boss_immune and bosses:
            self._idle_steps += 1
            idle_seconds = self._idle_steps / 15.0
            reward -= self.w_idle_penalty * idle_seconds

        # --- Damage taken by player (negative reward) ---
        player_max_hp = max(p.get("max_hp", 1), 1)
        player_hp_delta = rc.get("player_hp_delta", 0.0)
        got_hit = player_hp_delta < 0
        if got_hit:
            self._hits_taken += 1
            self._dodge_streak = 0
            reward -= self.w_player_dmg * (-player_hp_delta / player_max_hp)
            # Compounding hit penalty: each successive hit costs more
            # 1st hit = w, 2nd = 2w, 5th = 5w — punishes sloppy play harder over time
            if self.w_hit_compound > 0:
                reward -= self.w_hit_compound * self._hits_taken

        # --- Survival bonus ---
        if self.w_survival > 0:
            reward += self.w_survival

        # --- Distance penalty ---
        dist = rc.get("distance_to_boss", 0.0)
        if dist > self.dist_thresh:
            excess = dist - self.dist_thresh
            # Linear penalty
            reward -= self.w_dist * excess
            # Quadratic penalty (makes far distances much worse)
            if self.w_dist_sq > 0:
                reward -= self.w_dist_sq * (excess * excess)

        # --- Proximity bonus (within N blocks of boss) ---
        if self.w_proximity > 0 and dist <= self.proximity_radius:
            reward += self.w_proximity

        # --- Dodge streak bonus (close to boss without getting hit) ---
        if self.w_dodge_streak > 0 and not got_hit and dist <= self.dodge_streak_radius:
            self._dodge_streak += 1
            streak_seconds = self._dodge_streak / 15.0
            reward += self.w_dodge_streak * streak_seconds

        # --- No-hit incremental bonus ---
        # Per step: w * 0.001 / (1 + hits)^2 — dense signal throughout the fight.
        # Starts large and degrades quickly with each hit taken, rewarding clean play
        # at every step rather than just at episode end.
        if self.w_nohit > 0:
            reward += self.w_nohit * 0.001 / (1.0 + self._hits_taken) ** 2

        # Track player death via HP every step — catches the case where the mod
        # sends boss_killed=True before (or without) player_died=True.
        if p.get("hp", 0) <= 0 or rc.get("player_died", False):
            self._player_died_this_episode = True

        # --- Terminal: player died ---
        if rc.get("player_died", False):
            reward -= self.w_death

        # --- Terminal: boss killed ---
        # Only reward a kill if the player is confirmed alive this episode.
        if rc.get("boss_killed", False):
            logger.info(
                f"boss_killed received: player_died_flag={rc.get('player_died', False)}, "
                f"player_hp={p.get('hp', '?')}, died_this_ep={self._player_died_this_episode}"
            )
        if rc.get("boss_killed", False) and not self._player_died_this_episode:
            reward += self.w_kill
            if self.w_flawless > 0:
                reward += self.w_flawless / (1 + self._hits_taken)
            if self.w_nohit > 0:
                reward += self.w_nohit * math.exp(-self._hits_taken)
            if self.w_nohit_kill > 0 and self._hits_taken == 0:
                reward += self.w_nohit_kill

        # --- Terminal: boss despawned (kited too far) ---
        if rc.get("boss_despawned", False):
            reward -= self.w_despawn

        # --- Wasteful heal penalty ---
        if self.w_wasteful_heal > 0 and action.get("quick_heal", 0) == 1:
            hp_frac = p.get("hp", 0) / player_max_hp
            if hp_frac > 0.6:
                reward -= self.w_wasteful_heal

        return reward
