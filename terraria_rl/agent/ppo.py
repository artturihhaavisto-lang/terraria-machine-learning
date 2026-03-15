import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.action_space import ActionSpace
from agent.policy_network import PolicyNetwork
from agent.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class PPOAgent:
    """Proximal Policy Optimization agent for the Terraria environment.

    Implements:
      - Clipped surrogate objective
      - Clipped value function loss
      - Entropy bonus
      - Gradient clipping
      - Early stopping on KL divergence
      - Advantage normalization
      - Optional learning rate scheduling
    """

    def __init__(self, config: Dict[str, Any], obs_dim: int):
        self.config = config
        ppo_cfg = config["ppo"]
        train_cfg = config["training"]

        self.device = torch.device(
            train_cfg.get("device", "cpu")
            if torch.cuda.is_available()
            else "cpu"
        )

        # Hyper-parameters.
        self.gamma: float = ppo_cfg["gamma"]
        self.gae_lambda: float = ppo_cfg["gae_lambda"]
        self.clip_epsilon: float = ppo_cfg["clip_epsilon"]
        self.entropy_coef: float = ppo_cfg["entropy_coef"]
        self.value_loss_coef: float = ppo_cfg["value_loss_coef"]
        self.max_grad_norm: float = ppo_cfg["max_grad_norm"]
        self.epochs_per_update: int = ppo_cfg["epochs_per_update"]
        self.minibatch_size: int = ppo_cfg["minibatch_size"]
        self.rollout_length: int = ppo_cfg["rollout_length"]
        self.normalize_advantages: bool = ppo_cfg["normalize_advantages"]
        self.target_kl: float = ppo_cfg.get("target_kl", 0.015)

        # Network and optimizer.
        self.policy_network = PolicyNetwork(config, obs_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=ppo_cfg["learning_rate"],
            eps=1e-5,
        )

        # Optional linear LR decay.
        self.total_timesteps: int = train_cfg.get("total_timesteps", 10_000_000)
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=int(self.total_timesteps / self.rollout_length),
        )

        self.action_space = ActionSpace(device=self.device)

        self.rollout_buffer = RolloutBuffer(
            rollout_length=self.rollout_length,
            obs_dim=obs_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[Dict[str, Any], float, float]:
        """Select an action given a normalized observation.

        Args:
            obs: Normalized observation array of shape (obs_dim,).
            deterministic: If True, take the mode instead of sampling.

        Returns:
            actions: Dict of action tensors.
            log_prob: Scalar log-probability.
            value: Critic's value estimate.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        binary_logits, aim_mean, aim_log_std, value = self.policy_network(obs_t)

        if deterministic:
            # Take Bernoulli mode (threshold 0) and Normal mean.
            binary_samples = (binary_logits > 0).float()
            aim_sample = torch.tanh(aim_mean) * np.pi
            actions = self.action_space._split_binary(binary_samples)
            actions["aim_angle"] = aim_sample
            log_prob = 0.0
        else:
            actions, log_prob_t, _ = self.action_space.sample_action(
                binary_logits, aim_mean, aim_log_std
            )
            log_prob = log_prob_t.item()

        return actions, log_prob, value.item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_gae(self, last_obs: np.ndarray, last_done: bool) -> None:
        """Bootstrap the last value and compute GAE over the current rollout.

        Args:
            last_obs: The observation following the final stored step.
            last_done: Whether the final stored step ended an episode.
        """
        obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value = self.policy_network(obs_t)
        self.rollout_buffer.compute_gae(
            last_value=last_value.item(), last_done=last_done
        )

    def train(self) -> Dict[str, float]:
        """Run PPO update on the current rollout buffer.

        Returns:
            Dict of mean training metrics across all epochs/minibatches.
        """
        policy_losses: list = []
        value_losses: list = []
        entropies: list = []
        approx_kls: list = []
        per_action_entropies: Dict[str, list] = {}

        for epoch in range(self.epochs_per_update):
            early_stop = False
            for batch in self.rollout_buffer.sample(self.minibatch_size):
                obs = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Normalize advantages.
                if self.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass.
                binary_logits, aim_mean, aim_log_std, values = self.policy_network(obs)

                log_probs, entropy, per_ent = self.action_space.evaluate_actions(
                    binary_logits, aim_mean, aim_log_std, actions
                )

                # Policy loss (clipped surrogate).
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped).
                value_pred_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (values - returns) ** 2
                value_loss_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus.
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Record metrics.
                with torch.no_grad():
                    approx_kl = ((old_log_probs - log_probs).mean()).item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl)

                for k, v in per_ent.items():
                    per_action_entropies.setdefault(k, []).append(
                        v.item() if isinstance(v, torch.Tensor) else v
                    )

                # Early stopping on KL divergence.
                if approx_kl > self.target_kl:
                    logger.debug(
                        f"Early stop at epoch {epoch} — approx_kl {approx_kl:.4f} > {self.target_kl}"
                    )
                    early_stop = True
                    break

            if early_stop:
                break

        self.lr_scheduler.step()
        self.rollout_buffer.reset()

        mean_per_ent = {k: float(np.mean(v)) for k, v in per_action_entropies.items()}

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "action_entropies": mean_per_ent,
        }
