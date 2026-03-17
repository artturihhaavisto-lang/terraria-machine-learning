"""
Gymnasium environment wrapping the Terraria BossMLMod TCP interface.

This env is fully boss-agnostic. It receives raw game state over TCP,
normalizes it into a fixed-size observation, computes a shaped reward,
and sends discrete actions back. The agent learns to fight any boss
from the same universal observation/reward signals.

Usage:
    env = TerrariaEnv(config)
    obs, info = env.reset()
    while True:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import logging
import numpy as np
import gymnasium as gym

from src.comm.tcp_client import TerrariaClient
from src.env.observation import make_observation_space, process_state
from src.env.action import make_action_space, numpy_to_action_dict, noop_action
from src.env.reward import RewardCalculator

logger = logging.getLogger(__name__)


class TerrariaEnv(gym.Env):
    """
    Gymnasium environment for Terraria boss fights.

    Observation: float32 vector (size depends on config, default 223)
    Action: MultiDiscrete([3, 2, 2, 16, 2, 2, 10]) (see action.py)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        conn_cfg = config.get("connection", {})
        obs_cfg = config.get("observation", {})
        act_cfg = config.get("action", {})
        reward_cfg = config.get("reward", {})
        ep_cfg = config.get("episode", {})

        # Spaces — observation size is computed from config
        self.observation_space = make_observation_space(obs_cfg)
        self.action_space = make_action_space(act_cfg)

        # TCP client
        self._client = TerrariaClient(
            host=conn_cfg.get("host", "127.0.0.1"),
            port=conn_cfg.get("port", 7777),
            connect_timeout=conn_cfg.get("connect_timeout", 60),
            recv_timeout=conn_cfg.get("recv_timeout", 5.0),
        )

        # Reward calculator (with hot-reload support)
        config_path = config.get("_config_path")
        self._reward_calc = RewardCalculator(reward_cfg, config_path=config_path)

        # Observation config (passed to process_state)
        self._obs_cfg = obs_cfg

        # Episode limits
        self._max_steps = ep_cfg.get("max_steps", 18000)
        self._reward_scale = ep_cfg.get("reward_scale", 1.0)

        # State tracking
        self._step_count = 0
        self._last_state: dict | None = None
        self._episode_reward = 0.0
        self._connected = False

        # Metadata for model save/restore
        self._loadout: dict | None = None
        self._boss_type: int = config.get("boss_type", 0)
        self._loadout_requested = False
        self._pending_set_loadout: dict | None = None

    def _ensure_connected(self) -> None:
        """Connect to the game, reconnecting if the connection was lost."""
        if not self._connected or not self._client.connected:
            self._client.close()  # Clean up any stale socket
            self._client.connect()
            self._connected = True

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment for a new episode.

        The game mod handles the actual reset (heal player, respawn boss)
        via auto-reset or /bml reset command. This method:
          1. Connects to the game if not already connected
          2. Waits for a non-done state (the start of a new episode)
          3. Returns the initial observation
        """
        super().reset(seed=seed)

        self._ensure_connected()
        self._reward_calc.reset()
        self._step_count = 0
        self._episode_reward = 0.0

        # The game auto-resets and sends a new state with done=False.
        # If we're receiving a done=True state (from previous episode end),
        # send no-op actions until we get the fresh episode state.
        state = self._recv_until_new_episode()
        self._last_state = state

        # Sync boss type from game state so train.py sees the actual boss immediately
        if state.get("boss_type", 0) > 0:
            self._boss_type = state["boss_type"]

        obs = process_state(state, self._obs_cfg)
        info = self._make_info(state)
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one step: send action to game, receive new state.

        Args:
            action: np.ndarray of shape (7,) from the policy.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action_dict = numpy_to_action_dict(action)

        # Attach any pending message from hot-reload
        if self._reward_calc.pending_message:
            action_dict["message"] = self._reward_calc.pending_message
            self._reward_calc.pending_message = None

        # Request loadout on first step if we don't have one yet
        if not self._loadout_requested and self._loadout is None:
            action_dict["request_loadout"] = True
            self._loadout_requested = True

        # Send loadout to game if pending (from model resume)
        if self._pending_set_loadout is not None:
            action_dict["set_loadout"] = self._pending_set_loadout
            self._pending_set_loadout = None

        # Send action, receive new state
        try:
            state = self._client.exchange(action_dict)
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection issue during step: {e}")
            self._connected = False
            # Return terminal state so SB3 resets the episode
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -1.0, True, False, {"connection_lost": True}

        self._last_state = state
        self._step_count += 1

        # Capture loadout and boss_type from game state
        if state.get("loadout") is not None:
            self._loadout = state["loadout"]
            logger.info(f"Loadout captured: {len(self._loadout.get('inventory', []))} inv, "
                        f"{len(self._loadout.get('armor', []))} armor slots")
        if state.get("boss_type", 0) > 0:
            self._boss_type = state["boss_type"]

        # Check for reload signal from game
        if state.get("reload_config", False):
            self._reward_calc._last_mtime = 0.0  # force reload on next check
            self._reward_calc._try_hot_reload()
            logger.info("Config reload triggered by game")

        # Process observation
        obs = process_state(state, self._obs_cfg)

        # Compute reward
        reward = self._reward_calc.compute(state, action_dict)
        reward *= self._reward_scale
        self._episode_reward += reward

        # Terminal conditions
        done = state.get("done", False)
        rc = state.get("reward_components", {})
        terminated = (rc.get("boss_killed", False)
                      or rc.get("boss_despawned", False)
                      or rc.get("player_died", False))
        truncated = self._step_count >= self._max_steps

        # If the game says done but we haven't detected why, treat as terminated
        if done and not terminated and not truncated:
            terminated = True

        info = self._make_info(state)
        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_length"] = self._step_count
            info["boss_killed"] = rc.get("boss_killed", False) and not self._reward_calc._player_died_this_episode
            info["boss_despawned"] = rc.get("boss_despawned", False)
            info["player_died"] = rc.get("player_died", False)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up the TCP connection."""
        self._client.close()
        self._connected = False

    def _recv_until_new_episode(self) -> dict:
        """
        Drain any leftover done=True states and wait for a fresh episode.
        Sends no-op actions while waiting.
        """
        # 600 exchanges should cover any auto-reset delay (default 120 ticks = 2s)
        max_wait = 600
        for _ in range(max_wait):
            try:
                state = self._client.exchange(noop_action())
                if not state.get("done", True):
                    return state
            except TimeoutError:
                continue
            except ConnectionError:
                self._connected = False
                raise

        raise TimeoutError(
            "Timed out waiting for new episode from game. "
            "Is auto-reset enabled? (/bml reset to force)"
        )

    def get_metadata(self) -> dict:
        """Return metadata to save alongside the model."""
        return {
            "boss_type": self._boss_type,
            "loadout": self._loadout,
        }

    def set_loadout_on_connect(self, loadout: dict | None) -> None:
        """Queue a loadout to send to the game on next step."""
        if loadout:
            self._pending_set_loadout = loadout

    def _make_info(self, state: dict) -> dict:
        """Build the info dict from raw state."""
        rc = state.get("reward_components", {})
        return {
            "tick": state.get("tick", 0),
            "game_episode": state.get("episode", 0),
            "game_step": state.get("step", 0),
            "distance_to_boss": rc.get("distance_to_boss", 0.0),
        }
