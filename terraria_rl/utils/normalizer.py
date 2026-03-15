import numpy as np
from typing import Optional, Dict, Any


class RunningMeanStd:
    """Tracks running mean and variance using Welford's online algorithm.

    Used for observation normalization during training.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of observations.

        Args:
            x: Array of observations with shape (batch, *shape) or (*shape,).
        """
        if x.ndim == len(self.shape):
            # Single sample — add batch dim.
            x = x[np.newaxis]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observations to approximately zero mean and unit variance.

        Args:
            x: Raw observation array.

        Returns:
            Normalized array clipped to [-10, 10].
        """
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": int(self.count),
            "shape": list(self.shape),
            "epsilon": self.epsilon,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = int(state["count"])
        self.shape = tuple(state["shape"])
        self.epsilon = float(state["epsilon"])


class ObservationNormalizer:
    """Wraps RunningMeanStd to normalize environment observations.

    Can be disabled via config (obs_normalization: false).
    """

    def __init__(self, obs_dim: int, enabled: bool = True, epsilon: float = 1e-8):
        self.obs_dim = obs_dim
        self.enabled = enabled
        self.rms = RunningMeanStd(shape=(obs_dim,), epsilon=epsilon)

    def update_and_normalize(self, obs: np.ndarray) -> np.ndarray:
        """Update running statistics and return normalized observation."""
        if not self.enabled:
            return obs.astype(np.float32)
        self.rms.update(obs)
        return self.rms.normalize(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize without updating statistics (used at eval time)."""
        if not self.enabled:
            return obs.astype(np.float32)
        return self.rms.normalize(obs)

    def get_state(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "obs_dim": self.obs_dim,
            "rms": self.rms.get_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.enabled = state["enabled"]
        self.obs_dim = state["obs_dim"]
        self.rms.load_state(state["rms"])
