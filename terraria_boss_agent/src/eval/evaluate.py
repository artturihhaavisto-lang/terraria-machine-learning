"""
Evaluation / inference script. Runs a trained agent in deterministic mode
and records episode statistics.

Usage:
    python -m src.eval.evaluate --model logs/models/final_model.zip
    python -m src.eval.evaluate --model logs/models/final_model.zip --episodes 50
"""

import argparse
import logging
import os
import json
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from src.env.terraria_env import TerrariaEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def evaluate(
    model_path: str,
    config: dict,
    num_episodes: int = 20,
    deterministic: bool = True,
    output_file: str | None = None,
):
    """
    Run the trained agent for num_episodes and collect statistics.

    Args:
        model_path: Path to the saved .zip model.
        config: Full config dict.
        num_episodes: Number of episodes to evaluate.
        deterministic: Use deterministic (greedy) policy if True.
        output_file: Optional path to save results JSON.
    """
    t_cfg = config.get("training", {})

    env = DummyVecEnv([lambda: Monitor(TerrariaEnv(config))])

    frame_stack = t_cfg.get("frame_stack", 4)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    use_lstm = t_cfg.get("use_lstm", False)
    if use_lstm:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, env=env)
    else:
        model = PPO.load(model_path, env=env)

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Evaluating for {num_episodes} episodes (deterministic={deterministic})")

    results = []
    wins = 0
    deaths = 0
    total_reward = 0.0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_info = {}

        while not done:
            if use_lstm:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts,
                    deterministic=deterministic,
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0]
            ep_steps += 1
            done = dones[0]

            if done:
                ep_info = infos[0]
                if use_lstm:
                    episode_starts = np.ones((1,), dtype=bool)

        boss_killed = ep_info.get("boss_killed", False)
        player_died = ep_info.get("player_died", False)

        if boss_killed:
            wins += 1
        if player_died:
            deaths += 1
        total_reward += ep_reward

        result = {
            "episode": ep,
            "reward": float(ep_reward),
            "steps": ep_steps,
            "boss_killed": boss_killed,
            "player_died": player_died,
        }
        results.append(result)

        logger.info(
            f"Episode {ep + 1}/{num_episodes}: "
            f"reward={ep_reward:.2f}, steps={ep_steps}, "
            f"{'WIN' if boss_killed else 'LOSS'}"
        )

    env.close()

    # Summary
    win_rate = wins / num_episodes if num_episodes > 0 else 0.0
    avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
    avg_steps = np.mean([r["steps"] for r in results]) if results else 0.0

    summary = {
        "model": model_path,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "win_rate": win_rate,
        "death_rate": deaths / num_episodes if num_episodes > 0 else 0.0,
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "episodes": results,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Summary:")
    logger.info(f"  Win rate:    {win_rate:.1%} ({wins}/{num_episodes})")
    logger.info(f"  Avg reward:  {avg_reward:.2f}")
    logger.info(f"  Avg steps:   {avg_steps:.0f}")
    logger.info(f"{'='*50}")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Terraria boss agent")
    parser.add_argument("--model", required=True, help="Path to model .zip file")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(
        model_path=args.model,
        config=config,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
