"""
Thread-safe metrics storage for the Terraria RL training dashboard.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeMetrics:
    episode: int
    reward: float
    length: int
    result: str          # "win", "loss", "timeout"
    boss_hp_remaining: float
    damage_dealt: float
    damage_taken: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingUpdate:
    update_step: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    learning_rate: float
    timestep: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveState:
    player_hp: float = 500.0
    player_hp_max: float = 500.0
    boss_hp: float = 1.0
    boss_hp_max: float = 1.0
    current_action: int = 0
    action_name: str = "idle"
    player_x: float = 0.0
    player_y: float = 0.0
    boss_x: float = 0.0
    boss_y: float = 0.0
    projectiles: List[Dict[str, float]] = field(default_factory=list)
    tile_grid: List[List[int]] = field(default_factory=list)
    episode: int = 0
    timestep: int = 0
    episode_reward: float = 0.0
    training_state: str = "idle"  # "training", "paused", "eval", "idle"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# Human-readable action names (adjust indices to match your action space)
ACTION_NAMES = {
    0: "idle",
    1: "move_left",
    2: "move_right",
    3: "jump",
    4: "attack",
    5: "move_left_attack",
    6: "move_right_attack",
    7: "jump_attack",
    8: "move_left_jump",
    9: "move_right_jump",
    10: "move_left_jump_attack",
    11: "move_right_jump_attack",
}


class MetricsStore:
    """
    Thread-safe store for all training metrics and live game state.

    Holds ring buffers for episode history and training update history,
    plus the most-recent live game state snapshot.
    """

    def __init__(self, max_episodes: int = 10_000, max_updates: int = 10_000):
        self._lock = threading.RLock()

        # Ring buffers
        self._episodes: deque[EpisodeMetrics] = deque(maxlen=max_episodes)
        self._training_updates: deque[TrainingUpdate] = deque(maxlen=max_updates)

        # Live state
        self._live: LiveState = LiveState()

        # Global counters
        self._total_timesteps: int = 0
        self._best_reward: float = float("-inf")
        self._start_time: float = time.time()

        # Rolling win-rate window size
        self._win_rate_window: int = 100

        # Training log (recent messages)
        self._log: deque[Dict[str, Any]] = deque(maxlen=500)

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def add_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        result: str,
        boss_hp_remaining: float,
        damage_dealt: float,
        damage_taken: float,
    ) -> None:
        """Record metrics for a completed episode."""
        with self._lock:
            m = EpisodeMetrics(
                episode=episode,
                reward=reward,
                length=length,
                result=result,
                boss_hp_remaining=boss_hp_remaining,
                damage_dealt=damage_dealt,
                damage_taken=damage_taken,
            )
            self._episodes.append(m)
            if reward > self._best_reward:
                self._best_reward = reward
            self._log_event(
                "episode_end",
                f"Episode {episode} finished — reward: {reward:.2f}, "
                f"result: {result}, length: {length} ticks",
            )

    def add_training_update(
        self,
        update_step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl_divergence: float,
        learning_rate: float,
        timestep: int,
    ) -> None:
        """Record metrics from a PPO update step."""
        with self._lock:
            u = TrainingUpdate(
                update_step=update_step,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                kl_divergence=kl_divergence,
                learning_rate=learning_rate,
                timestep=timestep,
            )
            self._training_updates.append(u)
            self._total_timesteps = max(self._total_timesteps, timestep)

    def update_live_state(
        self,
        player_hp: Optional[float] = None,
        player_hp_max: Optional[float] = None,
        boss_hp: Optional[float] = None,
        boss_hp_max: Optional[float] = None,
        current_action: Optional[int] = None,
        player_x: Optional[float] = None,
        player_y: Optional[float] = None,
        boss_x: Optional[float] = None,
        boss_y: Optional[float] = None,
        projectiles: Optional[List[Dict[str, float]]] = None,
        tile_grid: Optional[List[List[int]]] = None,
        episode: Optional[int] = None,
        timestep: Optional[int] = None,
        episode_reward: Optional[float] = None,
        training_state: Optional[str] = None,
    ) -> None:
        """Update the current live game state (called every tick or second)."""
        with self._lock:
            s = self._live
            if player_hp is not None:
                s.player_hp = player_hp
            if player_hp_max is not None:
                s.player_hp_max = player_hp_max
            if boss_hp is not None:
                s.boss_hp = boss_hp
            if boss_hp_max is not None:
                s.boss_hp_max = boss_hp_max
            if current_action is not None:
                s.current_action = current_action
                s.action_name = ACTION_NAMES.get(current_action, f"action_{current_action}")
            if player_x is not None:
                s.player_x = player_x
            if player_y is not None:
                s.player_y = player_y
            if boss_x is not None:
                s.boss_x = boss_x
            if boss_y is not None:
                s.boss_y = boss_y
            if projectiles is not None:
                s.projectiles = projectiles
            if tile_grid is not None:
                s.tile_grid = tile_grid
            if episode is not None:
                s.episode = episode
            if timestep is not None:
                s.timestep = timestep
                self._total_timesteps = max(self._total_timesteps, timestep)
            if episode_reward is not None:
                s.episode_reward = episode_reward
            if training_state is not None:
                s.training_state = training_state
            s.timestamp = time.time()

    def set_training_state(self, state: str) -> None:
        """Set the high-level training state: 'training', 'paused', 'eval', 'idle'."""
        with self._lock:
            self._live.training_state = state
            self._log_event("state_change", f"Training state changed to: {state}")

    def log_event(self, event_type: str, message: str) -> None:
        """Public method to add a log entry."""
        with self._lock:
            self._log_event(event_type, message)

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a high-level summary suitable for the /api/status endpoint."""
        with self._lock:
            recent = list(self._episodes)[-self._win_rate_window :]
            win_rate = (
                sum(1 for e in recent if e.result == "win") / len(recent)
                if recent
                else 0.0
            )
            avg_reward = (
                sum(e.reward for e in recent) / len(recent) if recent else 0.0
            )
            elapsed = time.time() - self._start_time
            return {
                "episode": self._live.episode,
                "total_timesteps": self._total_timesteps,
                "win_rate": round(win_rate, 4),
                "avg_reward": round(avg_reward, 4),
                "best_reward": round(self._best_reward, 4)
                if self._best_reward != float("-inf")
                else None,
                "current_episode_reward": round(self._live.episode_reward, 4),
                "training_state": self._live.training_state,
                "elapsed_seconds": round(elapsed, 1),
                "total_episodes": len(self._episodes),
            }

    def get_recent_episodes(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the last *n* episode metric dicts."""
        with self._lock:
            episodes = list(self._episodes)
            return [e.to_dict() for e in episodes[-n:]]

    def get_recent_updates(self, n: int = 100) -> List[Dict[str, Any]]:
        """Return the last *n* training-update metric dicts."""
        with self._lock:
            updates = list(self._training_updates)
            return [u.to_dict() for u in updates[-n:]]

    def get_live_state(self) -> Dict[str, Any]:
        """Return the current live game state dict."""
        with self._lock:
            return self._live.to_dict()

    def get_recent_log(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last *n* log entries."""
        with self._lock:
            entries = list(self._log)
            return entries[-n:]

    def get_rolling_win_rate(self, window: int = 100) -> List[float]:
        """
        Return a list of rolling win rates (one per episode) computed over
        the given window.  Length equals total recorded episodes.
        """
        with self._lock:
            episodes = list(self._episodes)
        rates: List[float] = []
        for i, ep in enumerate(episodes):
            start = max(0, i + 1 - window)
            chunk = episodes[start : i + 1]
            rates.append(sum(1 for e in chunk if e.result == "win") / len(chunk))
        return rates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, message: str) -> None:
        """Append a timestamped log entry (must be called with lock held)."""
        self._log.append(
            {
                "type": event_type,
                "message": message,
                "timestamp": time.time(),
            }
        )
