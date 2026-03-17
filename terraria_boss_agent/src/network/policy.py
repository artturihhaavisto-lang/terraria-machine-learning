"""
Custom policy network for PPO.

Architecture: Two-layer MLP (256x256) with optional frame stacking.

Why MLP over CNN:
  - Our observation is a structured feature vector, not a spatial image.
  - MLP is simpler, faster, and sufficient for ~223-dimensional input.

Why frame stacking over LSTM:
  - SB3's standard PPO supports frame stacking out of the box (VecFrameStack).
  - LSTM requires sb3-contrib's RecurrentPPO, which has different API quirks.
  - Frame stacking with 4 frames captures short-term temporal patterns
    (boss movement direction, projectile trajectories, attack timing).
  - LSTM is available as an option via config (use_lstm: true).

For the default MLP policy, we use SB3's built-in MlpPolicy with custom
net_arch. No custom feature extractor needed since the observation is
already a flat normalized vector.
"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym


class TerrariaFeaturesExtractor(BaseFeaturesExtractor):
    """
    Optional custom feature extractor that splits the observation into
    semantic groups (player, boss, projectiles) and processes them with
    separate sub-networks before concatenating.

    This can help the network learn group-specific representations, but
    the default flat MLP works well enough for initial training.

    Enable via policy_kwargs=dict(features_extractor_class=TerrariaFeaturesExtractor).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 max_buffs: int = 22,
                 max_segments: int = 10,
                 max_projectiles: int = 20):
        super().__init__(observation_space, features_dim)

        # Compute layout boundaries from config-driven sizes
        player_end = PLAYER_FEATURES + max_buffs * BUFF_FEATURES_PER_SLOT + ITEM_FEATURES
        boss_end = player_end + BOSS_FEATURES
        seg_end = boss_end + max_segments * SEGMENT_FEATURES
        proj_end = seg_end + max_projectiles * PROJECTILE_FEATURES
        env_end = proj_end + ENV_FEATURES

        self._player_end = player_end
        self._seg_end = seg_end
        self._proj_end = proj_end
        self._env_end = env_end

        player_size = player_end
        boss_size = seg_end - player_end
        proj_size = proj_end - seg_end
        env_size = env_end - proj_end

        self.player_net = nn.Sequential(
            nn.Linear(player_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.boss_net = nn.Sequential(
            nn.Linear(boss_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.proj_net = nn.Sequential(
            nn.Linear(proj_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.env_net = nn.Sequential(
            nn.Linear(env_size, 16),
            nn.ReLU(),
        )

        # Fusion: 32 + 32 + 32 + 16 = 112 -> features_dim
        self.fusion = nn.Sequential(
            nn.Linear(112, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        player_feats = observations[:, :self._player_end]
        boss_feats = observations[:, self._player_end:self._seg_end]
        proj_feats = observations[:, self._seg_end:self._proj_end]
        env_feats = observations[:, self._proj_end:self._env_end]

        p = self.player_net(player_feats)
        b = self.boss_net(boss_feats)
        pr = self.proj_net(proj_feats)
        e = self.env_net(env_feats)

        combined = torch.cat([p, b, pr, e], dim=1)
        return self.fusion(combined)


def get_policy_kwargs(cfg: dict) -> dict:
    """
    Build policy_kwargs for SB3's PPO based on training config.

    Args:
        cfg: Training config section.

    Returns:
        Dict suitable for PPO(policy_kwargs=...).
    """
    policy_layers = cfg.get("policy_layers", [256, 256])
    value_layers = cfg.get("value_layers", [256, 256])

    kwargs: dict = {
        "net_arch": dict(pi=policy_layers, vf=value_layers),
    }

    return kwargs
