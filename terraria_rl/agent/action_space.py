from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np


class ActionSpace:
    """Encodes and decodes actions between network distributions and JSON wire format.

    Action components:
      - movement (4): left, right, up, down — Bernoulli per bit
      - jump (1): Bernoulli
      - use_item (1): Bernoulli
      - dash (1): Bernoulli
      - grapple (1): Bernoulli
      - heal (1): Bernoulli
      - aim_angle (1): Normal (continuous, mapped to [-π, π])

    Total: 9 Bernoulli + 1 Normal = 10 action dimensions.
    """

    BINARY_KEYS = ["movement", "jump", "use_item", "dash", "grapple", "heal"]
    BINARY_SIZES = [4, 1, 1, 1, 1, 1]  # must sum to 9

    def __init__(self, device: torch.device):
        self.device = device

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_action(
        self,
        binary_logits: torch.Tensor,
        aim_mean: torch.Tensor,
        aim_log_std: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample actions from the policy distributions.

        Args:
            binary_logits: Tensor of shape (batch, 9) — raw logits for Bernoulli.
            aim_mean: Tensor of shape (batch, 1) — mean of aim Normal dist.
            aim_log_std: Tensor of shape (batch, 1) — log std of aim Normal dist.

        Returns:
            actions: Dict mapping action names to tensors.
            log_prob: Sum of log-probs across all action components, shape (batch,).
            distributions: Dict of distribution objects for entropy computation.
        """
        # Bernoulli distributions.
        binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
        binary_samples = binary_dist.sample()  # (batch, 9)
        binary_log_prob = binary_dist.log_prob(binary_samples).sum(dim=-1)  # (batch,)

        # Normal distribution for aim angle.
        aim_std = torch.exp(aim_log_std.clamp(-4.0, 2.0))
        aim_dist = torch.distributions.Normal(aim_mean, aim_std)
        aim_sample = aim_dist.rsample()  # (batch, 1)
        # Squash to [-π, π] via tanh.
        aim_squashed = torch.tanh(aim_sample) * np.pi
        # Log-prob with tanh correction.
        aim_log_prob = aim_dist.log_prob(aim_sample) - torch.log(
            1 - aim_squashed.pow(2) / (np.pi ** 2) + 1e-6
        )
        aim_log_prob = aim_log_prob.sum(dim=-1)  # (batch,)

        total_log_prob = binary_log_prob + aim_log_prob

        # Split binary samples into named components.
        actions = self._split_binary(binary_samples)
        actions["aim_angle"] = aim_squashed
        distributions = {
            "binary": binary_dist,
            "aim_angle": aim_dist,
        }
        return actions, total_log_prob, distributions

    def evaluate_actions(
        self,
        binary_logits: torch.Tensor,
        aim_mean: torch.Tensor,
        aim_log_std: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate log-probs and entropy of given actions under current policy.

        Args:
            binary_logits: (batch, 9)
            aim_mean: (batch, 1)
            aim_log_std: (batch, 1)
            actions: Dict of action tensors sampled earlier.

        Returns:
            log_prob: (batch,) total log-prob.
            entropy: (batch,) total entropy.
            per_action_entropy: Dict of mean scalar entropies per head.
        """
        # Reconstruct binary action tensor.
        binary_actions = self._join_binary(actions)  # (batch, 9)

        binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
        binary_log_prob = binary_dist.log_prob(binary_actions).sum(dim=-1)
        binary_entropy = binary_dist.entropy().sum(dim=-1)

        aim_std = torch.exp(aim_log_std.clamp(-4.0, 2.0))
        aim_dist = torch.distributions.Normal(aim_mean, aim_std)

        # Recover pre-squash sample from stored squashed action.
        squashed = actions["aim_angle"]
        pre_squash = torch.atanh((squashed / np.pi).clamp(-0.9999, 0.9999))
        aim_log_prob = aim_dist.log_prob(pre_squash) - torch.log(
            1 - squashed.pow(2) / (np.pi ** 2) + 1e-6
        )
        aim_log_prob = aim_log_prob.sum(dim=-1)
        aim_entropy = aim_dist.entropy().sum(dim=-1)

        total_log_prob = binary_log_prob + aim_log_prob
        total_entropy = binary_entropy + aim_entropy

        per_action_entropy: Dict[str, torch.Tensor] = {}
        offset = 0
        for key, size in zip(self.BINARY_KEYS, self.BINARY_SIZES):
            per_action_entropy[key] = binary_dist.entropy()[:, offset:offset + size].mean()
            offset += size
        per_action_entropy["aim_angle"] = aim_entropy.mean()

        return total_log_prob, total_entropy, per_action_entropy

    # ------------------------------------------------------------------
    # Encode / decode for storage
    # ------------------------------------------------------------------

    def encode_action(self, actions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert tensors to JSON-compatible Python dicts/lists."""
        encoded: Dict[str, Any] = {}
        for key in self.BINARY_KEYS + ["aim_angle"]:
            val = actions[key]
            if val.dim() == 0:
                encoded[key] = val.item()
            else:
                encoded[key] = val.cpu().tolist()
        return encoded

    def decode_action(
        self, action_dict: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Convert JSON action dict back to tensors on self.device."""
        decoded: Dict[str, torch.Tensor] = {}
        for key in self.BINARY_KEYS:
            decoded[key] = torch.tensor(
                action_dict[key], dtype=torch.float32, device=self.device
            )
        decoded["aim_angle"] = torch.tensor(
            action_dict["aim_angle"], dtype=torch.float32, device=self.device
        )
        if decoded["aim_angle"].dim() == 0:
            decoded["aim_angle"] = decoded["aim_angle"].unsqueeze(0)
        return decoded

    def actions_to_gym_dict(
        self, actions: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Convert action tensors to numpy arrays compatible with TerrariaEnv action space."""
        result: Dict[str, np.ndarray] = {}
        for key in self.BINARY_KEYS:
            result[key] = actions[key].cpu().numpy().astype(np.int8)
        result["aim_angle"] = actions["aim_angle"].cpu().numpy().astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_binary(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a (batch, 9) binary tensor into named components."""
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for key, size in zip(self.BINARY_KEYS, self.BINARY_SIZES):
            result[key] = tensor[:, offset:offset + size]
            offset += size
        return result

    def _join_binary(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate named binary components back to (batch, 9)."""
        parts = []
        for key in self.BINARY_KEYS:
            t = actions[key]
            if t.dim() == 1:
                t = t.unsqueeze(1)
            parts.append(t)
        return torch.cat(parts, dim=-1)
