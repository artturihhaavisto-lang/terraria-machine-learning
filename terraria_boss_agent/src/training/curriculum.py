"""
Curriculum learning: progress through bosses by difficulty.

The curriculum is defined as an ordered list of boss NPC type IDs in
default.yaml. No boss-specific reward tuning — the same reward function
applies to all bosses. The curriculum only controls WHICH boss is spawned.

Progression rule: advance to the next boss when the agent achieves
win_rate_threshold over the last min_episodes_per_stage episodes.

Communication with the game: the curriculum manager sends a JSON command
over TCP to change the boss type. The mod's /bml boss <id> command
equivalent is sent as a control message.
"""

import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that tracks win rate and advances the boss stage.

    When the agent reaches the win_rate_threshold on the current boss,
    it sends a boss-change command to the game via the environment's
    TCP client and moves to the next stage.
    """

    def __init__(self, cfg: dict, verbose: int = 0):
        super().__init__(verbose)
        self.stages: list[dict] = cfg.get("stages", [])
        self.win_rate_threshold: float = cfg.get("win_rate_threshold", 0.5)
        self.min_episodes: int = cfg.get("min_episodes_per_stage", 100)
        self.god_mode_episodes: int = cfg.get("god_mode_episodes", 0)

        self.current_stage: int = 0
        self.episode_outcomes: list[bool] = []
        self.total_episodes: int = 0

    def _on_step(self) -> bool:
        if not self.stages:
            return True

        for info in self.locals.get("infos", []):
            if "boss_killed" not in info:
                continue

            self.episode_outcomes.append(info["boss_killed"])
            self.total_episodes += 1

            # Check if we should advance
            if len(self.episode_outcomes) >= self.min_episodes:
                recent = self.episode_outcomes[-self.min_episodes:]
                win_rate = sum(recent) / len(recent)

                self.logger.record("curriculum/stage", self.current_stage)
                self.logger.record("curriculum/win_rate", win_rate)

                if self.current_stage < len(self.stages):
                    boss_type = self.stages[self.current_stage].get("boss_type", 50)
                    self.logger.record("curriculum/boss_type", boss_type)

                if win_rate >= self.win_rate_threshold:
                    self._advance_stage()

        return True

    def _advance_stage(self) -> None:
        """Move to the next boss in the curriculum."""
        self.current_stage += 1
        self.episode_outcomes.clear()

        if self.current_stage >= len(self.stages):
            logger.info("Curriculum complete! All stages cleared.")
            return

        next_boss = self.stages[self.current_stage].get("boss_type", 50)
        logger.info(
            f"Curriculum advancing to stage {self.current_stage}: "
            f"boss type {next_boss}"
        )

        # Send boss change command to the game.
        # Access the environment through the training locals.
        # The VecEnv wraps our TerrariaEnv; we reach into it to send
        # a control message via the TCP client.
        try:
            vec_env = self.training_env
            if vec_env is not None:
                # DummyVecEnv stores envs in .envs attribute
                for env in getattr(vec_env, "envs", []):
                    # Unwrap Monitor if needed
                    inner = env
                    while hasattr(inner, "env"):
                        inner = inner.env
                    if hasattr(inner, "_client") and inner._client.connected:
                        # Send a special control action that the mod interprets
                        # as a boss change. We use a reserved field for this.
                        # The mod's BossMLSystem reads the "boss_type" field
                        # if present in the action packet and updates config.
                        inner._client.send_action({
                            "move": 1, "jump": 0, "use_item": 0,
                            "aim_sector": 0, "hook": 0, "quick_heal": 0,
                            "set_boss_type": next_boss,
                        })
                        logger.info(f"Sent boss type change to game: {next_boss}")
        except Exception as e:
            logger.warning(f"Failed to send boss change to game: {e}")

    def _on_training_start(self) -> None:
        """Log the initial curriculum stage."""
        if self.stages:
            boss_type = self.stages[0].get("boss_type", 50)
            logger.info(f"Curriculum starting at stage 0: boss type {boss_type}")
