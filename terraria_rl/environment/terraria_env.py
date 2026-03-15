import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from environment.observation_parser import ObservationParser
from environment.reward_shaper import RewardShaper
from environment.socket_client import SocketClient

logger = logging.getLogger(__name__)


class TerrariaEnv(gym.Env):
    """Gymnasium-compatible environment that communicates with the tModLoader mod.

    Protocol:
      - The mod sends a JSON observation every game tick.
      - Python responds with a JSON action dict.
      - Episode boundaries are signalled via 'episode_end' messages.

    Observation space: Box(obs_dim * frame_stack,) float32
    Action space: Dict of binary and continuous action components.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        env_cfg = config["environment"]
        obs_cfg = config.get("observation", {})

        self.max_episode_ticks: int = env_cfg["max_episode_ticks"]
        self.frame_stack: int = obs_cfg.get("frame_stack", 1)

        self._parser = ObservationParser(config)
        self._shaper = RewardShaper(config)
        self._client = SocketClient(
            host=env_cfg["host"],
            port=env_cfg["port"],
        )

        # Observation space.
        stacked_dim = self._parser.stacked_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(stacked_dim,),
            dtype=np.float32,
        )

        # Action space — Dict with binary and continuous components.
        self.action_space = gym.spaces.Dict({
            "movement": gym.spaces.MultiBinary(4),   # left, right, up, down
            "jump": gym.spaces.MultiBinary(1),
            "use_item": gym.spaces.MultiBinary(1),
            "dash": gym.spaces.MultiBinary(1),
            "grapple": gym.spaces.MultiBinary(1),
            "heal": gym.spaces.MultiBinary(1),
            "aim_angle": gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
            ),
        })

        self._prev_raw_obs: Optional[Dict[str, Any]] = None
        self._current_tick: int = 0
        self._episode_reward: float = 0.0
        self._connected: bool = False

        # Episode stats for logging.
        self.last_episode_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if not self._connected:
            self._client.connect()
            self._connected = True

        self._parser.reset_frame_stack()
        self._prev_raw_obs = None
        self._current_tick = 0
        self._episode_reward = 0.0

        # Send a reset signal and wait for the first observation.
        try:
            self._client.send({"type": "reset"})
            raw_obs = self._wait_for_observation()
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Reset failed: {e}")
            self._connected = False
            raise

        obs = self._parser.parse(raw_obs)
        self._prev_raw_obs = raw_obs
        info: Dict[str, Any] = {"tick": 0}
        return obs, info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._current_tick += 1

        # Build and send action JSON.
        action_json = self._build_action_json(action)
        try:
            self._client.send(action_json)
            raw_msg = self._client.receive()
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Step communication error: {e}")
            # Return a terminal step with a large penalty on comm error.
            obs = np.zeros(self._parser.stacked_obs_dim(), dtype=np.float32)
            return obs, -50.0, True, False, {"error": str(e)}

        # Parse the message — it may be a regular observation or episode_end.
        msg_type = raw_msg.get("type", "observation")
        terminated = False
        truncated = False
        episode_event: Optional[str] = None

        if msg_type == "episode_end":
            episode_event = raw_msg.get("reason", "unknown")
            terminated = episode_event in ("boss_killed", "player_death")
            truncated = episode_event == "timeout"
            # Use the embedded observation if present, else keep prev.
            raw_obs = raw_msg.get("finalObservation", self._prev_raw_obs or {})
        else:
            raw_obs = raw_msg

        # Timeout guard.
        if self._current_tick >= self.max_episode_ticks and not (terminated or truncated):
            episode_event = "timeout"
            truncated = True

        obs = self._parser.parse(raw_obs)
        reward = self._shaper.compute_reward(self._prev_raw_obs, raw_obs, episode_event)
        self._episode_reward += reward

        # Build info dict.
        info: Dict[str, Any] = {
            "tick": self._current_tick,
            "episode_event": episode_event,
        }
        if terminated or truncated:
            boss_hp = sum(
                float(p.get("hp", 0)) for p in raw_obs.get("bossParts", [])
            )
            boss_max_hp = sum(
                max(float(p.get("maxHp", 1)), 1.0)
                for p in raw_obs.get("bossParts", [])
            ) or 1.0
            player = raw_obs.get("player", {})
            player_max_hp = max(float(player.get("maxHp", 500)), 1.0)
            damage_dealt_ratio = raw_msg.get("totalDamageDealt", 0.0) / boss_max_hp
            damage_taken_ratio = raw_msg.get("totalDamageTaken", 0.0) / player_max_hp

            info.update({
                "episode_reward": self._episode_reward,
                "episode_length": self._current_tick,
                "win": episode_event == "boss_killed",
                "boss_hp_remaining": boss_hp / boss_max_hp,
                "damage_dealt_ratio": damage_dealt_ratio,
                "damage_taken_ratio": damage_taken_ratio,
            })
            self.last_episode_info = info

        self._prev_raw_obs = raw_obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            try:
                self._client.send({"type": "close"})
            except Exception:
                pass
            self._client.disconnect()
            self._connected = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_observation(self) -> Dict[str, Any]:
        """Block until a non-episode-end JSON message is received."""
        while True:
            msg = self._client.receive()
            if msg.get("type") != "episode_end":
                return msg
            logger.debug("Discarding episode_end message received during reset.")

    @staticmethod
    def _build_action_json(action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Gymnasium action dict to the wire format expected by the mod."""
        movement = action["movement"]
        return {
            "type": "action",
            "moveLeft": bool(movement[0]),
            "moveRight": bool(movement[1]),
            "moveUp": bool(movement[2]),
            "moveDown": bool(movement[3]),
            "jump": bool(action["jump"][0]),
            "useItem": bool(action["use_item"][0]),
            "dash": bool(action["dash"][0]),
            "grapple": bool(action["grapple"][0]),
            "heal": bool(action["heal"][0]),
            "aimAngle": float(action["aim_angle"][0]),
        }
