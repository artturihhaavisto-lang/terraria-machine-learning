from typing import Any, Dict, Optional
import math


class RewardShaper:
    """Computes shaped reward signals from consecutive observations and episode events.

    Reward components:
      - Terminal: boss_killed, player_death, timeout
      - Damage dealt: positive reward scaled by fraction of boss max HP dealt
      - Damage taken: negative reward scaled by fraction of player max HP lost
      - Survival bonus: small reward per tick
      - Proximity shaping: reward for being near ideal combat range
      - Dodge bonus: reward when nearby projectiles exist but no damage taken
    """

    def __init__(self, config: Dict[str, Any]):
        r = config["reward"]
        self.boss_killed_reward: float = r["boss_killed"]
        self.player_death_reward: float = r["player_death"]
        self.timeout_reward: float = r["timeout"]
        self.damage_dealt_scale: float = r["damage_dealt_scale"]
        self.damage_taken_scale: float = r["damage_taken_scale"]
        self.survival_per_tick: float = r["survival_per_tick"]
        self.proximity_weight: float = r["proximity_weight"]
        self.ideal_combat_range: float = r["ideal_combat_range"]
        self.dodge_bonus: float = r["dodge_bonus"]
        self.dodge_threat_threshold: int = int(r["dodge_threat_threshold"])

    def compute_reward(
        self,
        prev_obs: Optional[Dict[str, Any]],
        curr_obs: Dict[str, Any],
        episode_event: Optional[str],
    ) -> float:
        """Compute the shaped reward for a single tick transition.

        Args:
            prev_obs: Raw JSON observation from the previous tick, or None if
                      this is the first step of the episode.
            curr_obs: Raw JSON observation from the current tick.
            episode_event: One of "boss_killed", "player_death", "timeout",
                           or None for a regular step.

        Returns:
            Scalar reward for this timestep.
        """
        reward = 0.0

        # ----- Terminal rewards -----
        if episode_event == "boss_killed":
            return self.boss_killed_reward
        elif episode_event == "player_death":
            return self.player_death_reward
        elif episode_event == "timeout":
            return self.timeout_reward

        # ----- Step-level rewards -----
        reward += self.survival_per_tick

        if prev_obs is None:
            return reward

        player_prev = prev_obs.get("player", {})
        player_curr = curr_obs.get("player", {})

        # Damage dealt (boss HP decreased).
        reward += self._damage_dealt_reward(prev_obs, curr_obs)

        # Damage taken (player HP decreased).
        reward += self._damage_taken_reward(player_prev, player_curr)

        # Proximity shaping.
        reward += self._proximity_reward(curr_obs)

        # Dodge bonus.
        reward += self._dodge_reward(player_prev, player_curr, curr_obs)

        return reward

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def _damage_dealt_reward(
        self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]
    ) -> float:
        """Reward proportional to boss HP damage dealt this tick."""
        prev_parts = prev_obs.get("bossParts", [])
        curr_parts = curr_obs.get("bossParts", [])

        prev_hp = sum(float(p.get("hp", 0)) for p in prev_parts)
        curr_hp = sum(float(p.get("hp", 0)) for p in curr_parts)
        delta = prev_hp - curr_hp  # positive if damage was dealt

        if delta <= 0:
            return 0.0

        # Normalize by total boss max HP.
        max_hp = sum(
            max(float(p.get("maxHp", 1)), 1.0) for p in (prev_parts or curr_parts)
        )
        if max_hp <= 0:
            return 0.0

        return (delta / max_hp) * self.damage_dealt_scale

    def _damage_taken_reward(
        self,
        player_prev: Dict[str, Any],
        player_curr: Dict[str, Any],
    ) -> float:
        """Penalty proportional to player HP lost this tick."""
        hp_prev = float(player_prev.get("hp", 0))
        hp_curr = float(player_curr.get("hp", 0))
        delta = hp_prev - hp_curr  # positive if damage was taken

        if delta <= 0:
            return 0.0

        max_hp = max(float(player_curr.get("maxHp", 500)), 1.0)
        return (delta / max_hp) * self.damage_taken_scale  # negative scale

    def _proximity_reward(self, curr_obs: Dict[str, Any]) -> float:
        """Small reward based on how close the player is to ideal_combat_range."""
        player = curr_obs.get("player", {})
        px = float(player.get("x", 0.0))
        py = float(player.get("y", 0.0))

        boss_parts = curr_obs.get("bossParts", [])
        if not boss_parts:
            return 0.0

        # Use the closest active boss part.
        min_dist = math.inf
        for part in boss_parts:
            if not part.get("isActive", True):
                continue
            bx = float(part.get("x", 0.0))
            by = float(part.get("y", 0.0))
            d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if d < min_dist:
                min_dist = d

        if math.isinf(min_dist):
            return 0.0

        # Reward is maximal at ideal_combat_range and falls off as a Gaussian.
        deviation = abs(min_dist - self.ideal_combat_range)
        sigma = self.ideal_combat_range * 0.5
        proximity_score = math.exp(-0.5 * (deviation / sigma) ** 2)
        return self.proximity_weight * proximity_score

    def _dodge_reward(
        self,
        player_prev: Dict[str, Any],
        player_curr: Dict[str, Any],
        curr_obs: Dict[str, Any],
    ) -> float:
        """Bonus when many projectiles are nearby but no damage was taken."""
        hp_prev = float(player_prev.get("hp", 0))
        hp_curr = float(player_curr.get("hp", 0))
        took_damage = hp_curr < hp_prev

        if took_damage:
            return 0.0

        player = curr_obs.get("player", {})
        px = float(player.get("x", 0.0))
        py = float(player.get("y", 0.0))
        threat_radius = 200.0  # pixels

        nearby_threats = 0
        for proj in curr_obs.get("projectiles", []):
            if not proj.get("hostile", True):
                continue
            rx = float(proj.get("x", 0.0)) - px
            ry = float(proj.get("y", 0.0)) - py
            if math.sqrt(rx ** 2 + ry ** 2) < threat_radius:
                nearby_threats += 1

        if nearby_threats >= self.dodge_threat_threshold:
            return self.dodge_bonus * nearby_threats

        return 0.0
