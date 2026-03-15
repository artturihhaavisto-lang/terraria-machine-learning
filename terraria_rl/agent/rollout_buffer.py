from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores trajectory data for one PPO rollout and computes GAE advantages.

    Stores:
      - observations
      - actions (dict of tensors per action key)
      - log_probs
      - rewards
      - values
      - dones (episode termination flags)
      - advantages (computed via GAE)
      - returns (advantage + value)

    After a full rollout, call compute_gae() and then iterate via sample().
    """

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._ptr = 0
        self._full = False

        # Pre-allocate storage arrays on CPU.
        self.observations = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

        # Action storage — list of dicts (converted to tensors on demand).
        self.actions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Storing transitions
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: Dict[str, Any],
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Observation array.
            action: Dict of action tensors/values.
            log_prob: Log-probability of the action under the current policy.
            reward: Scalar reward for this step.
            value: Value estimate V(s) from the critic.
            done: True if this step ended an episode.
        """
        idx = self._ptr
        self.observations[idx] = obs
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = float(done)

        # Store action as CPU numpy/python scalars.
        stored: Dict[str, Any] = {}
        for key, val in action.items():
            if isinstance(val, torch.Tensor):
                stored[key] = val.detach().cpu().numpy()
            else:
                stored[key] = val
        if idx < len(self.actions):
            self.actions[idx] = stored
        else:
            self.actions.append(stored)

        self._ptr = (self._ptr + 1) % self.rollout_length
        if self._ptr == 0:
            self._full = True

    def is_ready(self) -> bool:
        """Return True when the buffer holds exactly rollout_length steps."""
        return self._full or self._ptr == self.rollout_length

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """Compute Generalized Advantage Estimation in-place.

        Args:
            last_value: Bootstrap value V(s_{T+1}) from the critic.
            last_done: Whether the final step ended an episode.
        """
        last_gae = 0.0
        next_value = last_value * (1.0 - float(last_done))

        for t in reversed(range(self.rollout_length)):
            next_non_terminal = 1.0 - self.dones[t]
            if t == self.rollout_length - 1:
                next_v = next_value
            else:
                next_v = self.values[t + 1] * (1.0 - self.dones[t + 1])
            delta = self.rewards[t] + self.gamma * next_v * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, minibatch_size: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield randomly-shuffled minibatches from the stored rollout.

        Args:
            minibatch_size: Number of transitions per minibatch.

        Yields:
            Dict with keys: observations, actions, log_probs, advantages, returns, values.
        """
        n = self.rollout_length
        indices = np.random.permutation(n)

        obs_t = torch.tensor(self.observations, dtype=torch.float32, device=self.device)
        lp_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
        val_t = torch.tensor(self.values, dtype=torch.float32, device=self.device)

        # Build action tensors once.
        action_tensors = self._stack_actions()

        for start in range(0, n, minibatch_size):
            idx = torch.tensor(
                indices[start: start + minibatch_size], dtype=torch.long, device=self.device
            )
            mb_actions = {k: v[idx] for k, v in action_tensors.items()}
            yield {
                "observations": obs_t[idx],
                "actions": mb_actions,
                "log_probs": lp_t[idx],
                "advantages": adv_t[idx],
                "returns": ret_t[idx],
                "values": val_t[idx],
            }

    def reset(self) -> None:
        """Clear the buffer and reset the pointer."""
        self._ptr = 0
        self._full = False
        self.actions = []
        self.advantages[:] = 0.0
        self.returns[:] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_actions(self) -> Dict[str, torch.Tensor]:
        """Convert list-of-dicts to dict-of-tensors."""
        keys = list(self.actions[0].keys())
        stacked: Dict[str, torch.Tensor] = {}
        for key in keys:
            arr = np.stack([a[key] for a in self.actions[: self.rollout_length]], axis=0)
            stacked[key] = torch.tensor(arr, dtype=torch.float32, device=self.device)
        return stacked
