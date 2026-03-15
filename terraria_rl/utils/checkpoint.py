import os
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    agent: Any,
    optimizer: torch.optim.Optimizer,
    normalizer: Any,
    update_count: int,
    total_timesteps: int,
    config: Dict[str, Any],
) -> None:
    """Save a full training checkpoint.

    Args:
        path: File path to write the checkpoint (.pt).
        agent: PPOAgent instance (must have policy_network attribute).
        optimizer: Optimizer used during training.
        normalizer: ObservationNormalizer instance.
        update_count: Number of PPO updates completed so far.
        total_timesteps: Total environment steps collected so far.
        config: Full configuration dictionary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "policy_state_dict": agent.policy_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer_state": normalizer.get_state(),
        "update_count": update_count,
        "total_timesteps": total_timesteps,
        "config": config,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path} (update {update_count}, steps {total_timesteps})")


def load_checkpoint(
    path: str,
    agent: Any,
    optimizer: Optional[torch.optim.Optimizer],
    normalizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore state into provided objects.

    Args:
        path: File path of the checkpoint to load.
        agent: PPOAgent instance whose policy_network will be restored.
        optimizer: Optimizer to restore state into (None if eval-only).
        normalizer: ObservationNormalizer to restore state into.
        device: Device to map tensors onto.

    Returns:
        Dictionary with "update_count", "total_timesteps", and "config".
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    agent.policy_network.load_state_dict(checkpoint["policy_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "normalizer_state" in checkpoint:
        normalizer.load_state(checkpoint["normalizer_state"])

    update_count = checkpoint.get("update_count", 0)
    total_timesteps = checkpoint.get("total_timesteps", 0)
    config = checkpoint.get("config", {})
    logger.info(
        f"Checkpoint loaded from {path} (update {update_count}, steps {total_timesteps})"
    )
    return {
        "update_count": update_count,
        "total_timesteps": total_timesteps,
        "config": config,
    }


def checkpoint_path(checkpoint_dir: str, update_count: int) -> str:
    """Build a checkpoint file path for a given update count."""
    return os.path.join(checkpoint_dir, f"checkpoint_{update_count:06d}.pt")


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recently written checkpoint in a directory.

    Returns:
        Absolute path to the latest checkpoint, or None if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
