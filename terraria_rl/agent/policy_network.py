from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


class SharedBackbone(nn.Module):
    """Multi-layer MLP shared between actor and critic.

    Uses LayerNorm + activation after each linear layer (when use_layer_norm=True).
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(_make_activation(activation))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorHead(nn.Module):
    """Produces action distribution parameters from the shared backbone output.

    Outputs:
      - binary_logits: (batch, 9) — logits for 9 Bernoulli actions
      - aim_mean: (batch, 1) — mean for aim angle Normal
      - aim_log_std: (batch, 1) — learnable log std for aim angle
    """

    TOTAL_BINARY = 9  # 4 movement + 5 single-bit actions

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        act = _make_activation(activation)

        self.binary_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.TOTAL_BINARY),
        )

        self.aim_mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _make_activation(activation),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # map to [-1, 1]; will be scaled by π in action_space
        )

        # Learned log std for aim angle (not input-dependent).
        self.aim_log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binary_logits = self.binary_head(x)
        aim_mean = self.aim_mean_head(x) * 3.14159265  # scale to [-π, π]
        aim_log_std = self.aim_log_std.expand(x.shape[0], 1)
        return binary_logits, aim_mean, aim_log_std


class CriticHead(nn.Module):
    """Value function head: returns scalar V(s)."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _make_activation(activation),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


class PolicyNetwork(nn.Module):
    """Full Actor-Critic network for the Terraria RL agent.

    forward() returns (binary_logits, aim_mean, aim_log_std, value).
    """

    def __init__(self, config: Dict[str, Any], obs_dim: int):
        super().__init__()
        net_cfg = config["network"]
        obs_cfg = config.get("observation", {})

        frame_stack = obs_cfg.get("frame_stack", 1)
        input_dim = obs_dim * frame_stack

        shared_layers: List[int] = net_cfg["shared_layers"]
        actor_hidden: int = net_cfg["actor_hidden"]
        critic_hidden: int = net_cfg["critic_hidden"]
        activation: str = net_cfg.get("activation", "relu")
        use_layer_norm: bool = net_cfg.get("use_layer_norm", True)

        self.backbone = SharedBackbone(
            input_dim=input_dim,
            layer_sizes=shared_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        backbone_out = self.backbone.output_dim

        self.actor = ActorHead(backbone_out, actor_hidden, activation)
        self.critic = CriticHead(backbone_out, critic_hidden, activation)

        self._init_weights()

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy outputs and value estimate.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).

        Returns:
            binary_logits: (batch, 9)
            aim_mean: (batch, 1)
            aim_log_std: (batch, 1)
            value: (batch,)
        """
        features = self.backbone(obs)
        binary_logits, aim_mean, aim_log_std = self.actor(features)
        value = self.critic(features)
        return binary_logits, aim_mean, aim_log_std, value

    def _init_weights(self) -> None:
        """Orthogonal initialization with appropriate gain values."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "critic" in name:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
