import csv
import logging
import os
from collections import deque
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logs training metrics to TensorBoard and a CSV backup file.

    Tracks per-episode metrics (reward, length, win/loss, boss HP, damage ratio)
    and per-update metrics (losses, KL, entropy, learning rate).
    """

    def __init__(self, log_dir: str, csv_filename: str = "training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        # CSV backup
        csv_path = os.path.join(log_dir, csv_filename)
        self._csv_file = open(csv_path, "a", newline="", buffering=1)
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_path = csv_path

        # Rolling win rate over last 100 episodes.
        self._win_history: deque = deque(maxlen=100)
        self._episode_count = 0
        self._update_count = 0

        logger.info(f"TrainingLogger writing to {log_dir}")

    # ------------------------------------------------------------------
    # Episode-level logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        reward: float,
        length: int,
        win: bool,
        boss_hp_remaining: float,
        damage_dealt_ratio: float,
        damage_taken_ratio: float,
        global_step: int,
    ) -> None:
        """Log metrics for a completed episode.

        Args:
            reward: Total undiscounted episode return.
            length: Episode length in ticks.
            win: True if the boss was killed this episode.
            boss_hp_remaining: Boss HP at episode end (0 if killed).
            damage_dealt_ratio: damage_dealt / boss_max_hp.
            damage_taken_ratio: damage_taken / player_max_hp.
            global_step: Total environment steps collected.
        """
        self._episode_count += 1
        self._win_history.append(float(win))
        win_rate = sum(self._win_history) / len(self._win_history)

        self.writer.add_scalar("episode/reward", reward, global_step)
        self.writer.add_scalar("episode/length", length, global_step)
        self.writer.add_scalar("episode/win", float(win), global_step)
        self.writer.add_scalar("episode/win_rate_100", win_rate, global_step)
        self.writer.add_scalar("episode/boss_hp_remaining", boss_hp_remaining, global_step)
        self.writer.add_scalar("episode/damage_dealt_ratio", damage_dealt_ratio, global_step)
        self.writer.add_scalar("episode/damage_taken_ratio", damage_taken_ratio, global_step)

        row = {
            "type": "episode",
            "global_step": global_step,
            "episode": self._episode_count,
            "reward": reward,
            "length": length,
            "win": int(win),
            "win_rate_100": win_rate,
            "boss_hp_remaining": boss_hp_remaining,
            "damage_dealt_ratio": damage_dealt_ratio,
            "damage_taken_ratio": damage_taken_ratio,
        }
        self._write_csv(row)

        logger.info(
            f"Episode {self._episode_count} | "
            f"step={global_step} | "
            f"reward={reward:.2f} | "
            f"len={length} | "
            f"win={win} | "
            f"win_rate={win_rate:.2%}"
        )

    # ------------------------------------------------------------------
    # Update-level logging
    # ------------------------------------------------------------------

    def log_update(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        approx_kl: float,
        learning_rate: float,
        global_step: int,
        action_entropies: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log metrics for a PPO update step.

        Args:
            policy_loss: Clipped surrogate policy loss.
            value_loss: Value function loss.
            entropy: Mean policy entropy across the batch.
            approx_kl: Approximate KL divergence between old and new policy.
            learning_rate: Current learning rate.
            global_step: Total environment steps collected.
            action_entropies: Optional per-action-head entropy values.
        """
        self._update_count += 1

        self.writer.add_scalar("update/policy_loss", policy_loss, global_step)
        self.writer.add_scalar("update/value_loss", value_loss, global_step)
        self.writer.add_scalar("update/entropy", entropy, global_step)
        self.writer.add_scalar("update/approx_kl", approx_kl, global_step)
        self.writer.add_scalar("update/learning_rate", learning_rate, global_step)

        if action_entropies:
            for name, ent in action_entropies.items():
                self.writer.add_scalar(f"update/entropy_{name}", ent, global_step)

        row: Dict[str, Any] = {
            "type": "update",
            "global_step": global_step,
            "update": self._update_count,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "learning_rate": learning_rate,
        }
        if action_entropies:
            row.update({f"entropy_{k}": v for k, v in action_entropies.items()})
        self._write_csv(row)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_csv(self, row: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            fieldnames = sorted(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            if os.path.getsize(self._csv_path) == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(row)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self._csv_file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
