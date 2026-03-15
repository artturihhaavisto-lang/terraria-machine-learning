import numpy as np
from typing import Any, Dict, List, Optional

# Feature counts per entity type (must match observation_dim in config).
# Layout for default config (2125 total):
#   25 + (10 × 20) + (50 × 8) + (30 × 10) + (40 × 30 × 1)
#   = 25 + 200 + 400 + 300 + 1200 = 2125
PLAYER_FEATURES = 25
BOSS_PART_FEATURES = 20
PROJECTILE_FEATURES = 8
ENEMY_FEATURES = 10
TILE_FEATURES = 1  # per tile cell

MAX_DISTANCE = 2000.0  # pixels, used for position normalization


class ObservationParser:
    """Converts raw JSON observation dictionaries to fixed-size numpy arrays.

    The observation vector layout (total 2125 for default config):
        [0:25]    Player state (25 features)
        [25:225]  Boss parts (10 × 20 features, zero-padded)
        [225:625] Projectiles (50 × 8 features, zero-padded)
        [625:925] Hostile NPCs (30 × 10 features, zero-padded)
        [925:2125] Tile grid (40 × 30 = 1200 values)

    All positions are relative to the player and normalized.
    HP values are normalized to [0, 1].
    """

    def __init__(self, config: Dict[str, Any]):
        env_cfg = config["environment"]
        obs_cfg = config.get("observation", {})

        self.obs_dim: int = env_cfg["observation_dim"]
        self.max_boss_parts: int = env_cfg["max_boss_parts"]
        self.max_projectiles: int = env_cfg["max_projectiles"]
        self.max_enemies: int = env_cfg["max_enemies"]
        self.tile_w: int = env_cfg["tile_grid_width"]
        self.tile_h: int = env_cfg["tile_grid_height"]

        self.frame_stack: int = obs_cfg.get("frame_stack", 1)
        self._frame_buffer: List[np.ndarray] = []

        # Verify dimension consistency.
        expected = (
            PLAYER_FEATURES
            + self.max_boss_parts * BOSS_PART_FEATURES
            + self.max_projectiles * PROJECTILE_FEATURES
            + self.max_enemies * ENEMY_FEATURES
            + self.tile_w * self.tile_h * TILE_FEATURES
        )
        assert expected == self.obs_dim, (
            f"observation_dim mismatch: config says {self.obs_dim}, "
            f"computed {expected}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw: Dict[str, Any]) -> np.ndarray:
        """Parse a raw JSON observation dict to a numpy float32 array.

        Args:
            raw: JSON dictionary from the Terraria mod.

        Returns:
            Float32 array of shape (obs_dim * frame_stack,) if frame stacking
            is enabled, or (obs_dim,) otherwise.
        """
        obs = self._build_single_obs(raw)

        if self.frame_stack > 1:
            return self._apply_frame_stack(obs)
        return obs

    def reset_frame_stack(self) -> None:
        """Clear the frame buffer at the start of a new episode."""
        self._frame_buffer = []

    def stacked_obs_dim(self) -> int:
        return self.obs_dim * self.frame_stack

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_single_obs(self, raw: Dict[str, Any]) -> np.ndarray:
        player = raw.get("player", {})
        px = float(player.get("x", 0.0))
        py = float(player.get("y", 0.0))

        parts = [
            self._parse_player(player),
            self._parse_boss_parts(raw.get("bossParts", []), px, py),
            self._parse_projectiles(raw.get("projectiles", []), px, py),
            self._parse_enemies(raw.get("hostileNPCs", []), px, py),
            self._parse_tiles(raw.get("tileGrid", [])),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.obs_dim, (
            f"Built obs shape {obs.shape[0]} != expected {self.obs_dim}"
        )
        return obs

    def _parse_player(self, p: Dict[str, Any]) -> np.ndarray:
        """25 features: position, velocity, HP, mana, buffs, facing, etc."""
        hp = float(p.get("hp", 0))
        max_hp = max(float(p.get("maxHp", 500)), 1.0)
        mana = float(p.get("mana", 0))
        max_mana = max(float(p.get("maxMana", 200)), 1.0)
        vx = float(p.get("velocityX", 0.0))
        vy = float(p.get("velocityY", 0.0))
        on_ground = float(p.get("onGround", False))
        facing_right = float(p.get("facingRight", True))
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        heal_cd = float(p.get("healCooldown", 0)) / 3600.0
        dashes_left = float(p.get("dashesLeft", 0)) / 3.0
        grapple_active = float(p.get("grappleActive", False))
        item_cd = float(p.get("itemCooldown", 0)) / 60.0
        defense = float(p.get("defense", 0)) / 100.0
        attack_power = float(p.get("attackPower", 0)) / 200.0
        movement_speed = float(p.get("movementSpeed", 1.0))
        jump_speed = float(p.get("jumpSpeed", 5.0)) / 10.0
        gravity = float(p.get("gravityDir", 1.0))

        # Buffs / debuffs flags (6 features).
        buffs = p.get("activeBuffs", [])
        has_potion_sick = float(any(b == "PotionSickness" for b in buffs))
        has_on_fire = float(any(b == "OnFire" for b in buffs))
        has_cursed = float(any(b == "Cursed" for b in buffs))
        has_frozen = float(any(b == "Frozen" for b in buffs))
        has_poisoned = float(any(b == "Poisoned" for b in buffs))
        has_electrified = float(any(b == "Electrified" for b in buffs))

        features = np.array([
            hp / max_hp,            # 0: HP ratio
            mana / max_mana,        # 1: mana ratio
            np.clip(vx / 20.0, -1, 1),  # 2: velocity x (normalized)
            np.clip(vy / 20.0, -1, 1),  # 3: velocity y
            on_ground,              # 4
            facing_right,           # 5
            np.clip(x / 8000.0, -1, 1),  # 6: world x
            np.clip(y / 2400.0, -1, 1),  # 7: world y
            heal_cd,                # 8
            dashes_left,            # 9
            grapple_active,         # 10
            item_cd,                # 11
            defense,                # 12
            attack_power,           # 13
            movement_speed / 2.0,   # 14
            jump_speed,             # 15
            gravity,                # 16
            has_potion_sick,        # 17
            has_on_fire,            # 18
            has_cursed,             # 19
            has_frozen,             # 20
            has_poisoned,           # 21
            has_electrified,        # 22
            float(p.get("invincibilityFrames", 0)) / 60.0,  # 23
            float(p.get("isJumping", False)),               # 24
        ], dtype=np.float32)
        return features  # shape (25,)

    def _parse_boss_parts(
        self, parts: List[Dict[str, Any]], px: float, py: float
    ) -> np.ndarray:
        """max_boss_parts × 20 features, zero-padded.

        Feature layout per part (20 total):
          0: rel_x, 1: rel_y, 2: hp_ratio, 3: vx, 4: vy,
          5: width, 6: height, 7: phase, 8: is_active, 9: part_type,
          10: despawn_dist, 11: ai0, 12: ai1, 13: ai2, 14: ai3,
          15: dist, 16: is_invincible, 17: rotation, 18: scale, 19: target_dist
        """
        out = np.zeros((self.max_boss_parts, BOSS_PART_FEATURES), dtype=np.float32)
        for i, part in enumerate(parts[: self.max_boss_parts]):
            bx = float(part.get("x", 0.0)) - px
            by = float(part.get("y", 0.0)) - py
            hp = float(part.get("hp", 0))
            max_hp = max(float(part.get("maxHp", 1)), 1.0)
            vx = float(part.get("velocityX", 0.0))
            vy = float(part.get("velocityY", 0.0))
            width = float(part.get("width", 32)) / 256.0
            height = float(part.get("height", 32)) / 256.0
            phase = float(part.get("phase", 0)) / 5.0
            is_active = float(part.get("isActive", True))
            part_type = float(part.get("partType", 0)) / 10.0
            despawn_dist = float(part.get("despawnDistance", 0)) / MAX_DISTANCE
            ai0 = float(part.get("ai0", 0.0)) / 100.0
            ai1 = float(part.get("ai1", 0.0)) / 100.0
            ai2 = float(part.get("ai2", 0.0)) / 100.0
            ai3 = float(part.get("ai3", 0.0)) / 100.0
            dist = np.sqrt(bx ** 2 + by ** 2) / MAX_DISTANCE
            rotation = float(part.get("rotation", 0.0)) / np.pi  # normalized to [-1,1]
            scale = float(part.get("scale", 1.0))
            target_dist = float(part.get("targetDistance", 0.0)) / MAX_DISTANCE

            out[i] = [
                np.clip(bx / MAX_DISTANCE, -1, 1),   # 0
                np.clip(by / MAX_DISTANCE, -1, 1),   # 1
                hp / max_hp,                          # 2
                np.clip(vx / 20.0, -1, 1),           # 3
                np.clip(vy / 20.0, -1, 1),           # 4
                width,                                # 5
                height,                               # 6
                phase,                                # 7
                is_active,                            # 8
                part_type,                            # 9
                despawn_dist,                         # 10
                np.clip(ai0, -1, 1),                 # 11
                np.clip(ai1, -1, 1),                 # 12
                np.clip(ai2, -1, 1),                 # 13
                np.clip(ai3, -1, 1),                 # 14
                np.clip(dist, 0, 1),                 # 15
                float(part.get("isInvincible", False)),  # 16
                np.clip(rotation, -1, 1),            # 17
                np.clip(scale, 0, 3),                # 18
                np.clip(target_dist, 0, 1),          # 19
            ]
        return out.flatten()

    def _parse_projectiles(
        self, projectiles: List[Dict[str, Any]], px: float, py: float
    ) -> np.ndarray:
        """max_projectiles × 8 features, zero-padded."""
        out = np.zeros((self.max_projectiles, PROJECTILE_FEATURES), dtype=np.float32)
        for i, proj in enumerate(projectiles[: self.max_projectiles]):
            rx = float(proj.get("x", 0.0)) - px
            ry = float(proj.get("y", 0.0)) - py
            vx = float(proj.get("velocityX", 0.0))
            vy = float(proj.get("velocityY", 0.0))
            dmg = float(proj.get("damage", 0)) / 200.0
            dist = np.sqrt(rx ** 2 + ry ** 2) / MAX_DISTANCE
            hostile = float(proj.get("hostile", True))
            proj_type = float(proj.get("type", 0)) / 1000.0

            out[i] = [
                np.clip(rx / MAX_DISTANCE, -1, 1),
                np.clip(ry / MAX_DISTANCE, -1, 1),
                np.clip(vx / 20.0, -1, 1),
                np.clip(vy / 20.0, -1, 1),
                np.clip(dmg, 0, 1),
                np.clip(dist, 0, 1),
                hostile,
                proj_type,
            ]
        return out.flatten()

    def _parse_enemies(
        self, enemies: List[Dict[str, Any]], px: float, py: float
    ) -> np.ndarray:
        """max_enemies × 10 features, zero-padded."""
        out = np.zeros((self.max_enemies, ENEMY_FEATURES), dtype=np.float32)
        for i, npc in enumerate(enemies[: self.max_enemies]):
            rx = float(npc.get("x", 0.0)) - px
            ry = float(npc.get("y", 0.0)) - py
            hp = float(npc.get("hp", 0))
            max_hp = max(float(npc.get("maxHp", 1)), 1.0)
            vx = float(npc.get("velocityX", 0.0))
            vy = float(npc.get("velocityY", 0.0))
            dist = np.sqrt(rx ** 2 + ry ** 2) / MAX_DISTANCE
            npc_type = float(npc.get("type", 0)) / 1000.0
            damage = float(npc.get("damage", 0)) / 200.0
            can_fly = float(npc.get("noGravity", False))

            out[i] = [
                np.clip(rx / MAX_DISTANCE, -1, 1),
                np.clip(ry / MAX_DISTANCE, -1, 1),
                hp / max_hp,
                np.clip(vx / 20.0, -1, 1),
                np.clip(vy / 20.0, -1, 1),
                np.clip(dist, 0, 1),
                npc_type,
                np.clip(damage, 0, 1),
                can_fly,
                float(npc.get("isHostile", True)),
            ]
        return out.flatten()

    def _parse_tiles(self, tile_grid: Any) -> np.ndarray:
        """1200 values from a 40×30 tile grid.

        tile_grid may be:
          - A flat list of ints (tile type IDs)
          - A 2D list [[row0_col0, ...], ...]
          - An empty list (return zeros)
        """
        total = self.tile_w * self.tile_h
        out = np.zeros(total, dtype=np.float32)

        if not tile_grid:
            return out

        # Flatten if 2D.
        if tile_grid and isinstance(tile_grid[0], list):
            flat: List[int] = []
            for row in tile_grid:
                flat.extend(row)
        else:
            flat = tile_grid

        n = min(len(flat), total)
        # Normalize tile IDs (vanilla Terraria has ~700 tile types).
        out[:n] = np.clip(np.array(flat[:n], dtype=np.float32) / 700.0, 0.0, 1.0)
        return out

    def _apply_frame_stack(self, obs: np.ndarray) -> np.ndarray:
        """Maintain a rolling frame buffer and return the stacked observation."""
        self._frame_buffer.append(obs)
        if len(self._frame_buffer) > self.frame_stack:
            self._frame_buffer.pop(0)
        # Pad with zeros if not enough frames yet.
        while len(self._frame_buffer) < self.frame_stack:
            self._frame_buffer.insert(0, np.zeros_like(obs))
        return np.concatenate(self._frame_buffer)
