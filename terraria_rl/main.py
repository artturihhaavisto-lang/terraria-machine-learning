"""Entry point for the Terraria RL agent.

Usage:
    # Train from scratch:
    python main.py --config configs/default.yaml --mode train

    # Resume training from a checkpoint:
    python main.py --config configs/default.yaml --mode train --checkpoint checkpoints/checkpoint_000100.pt

    # Evaluate a saved checkpoint:
    python main.py --config configs/default.yaml --mode eval --checkpoint checkpoints/checkpoint_000100.pt
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# Add the terraria_rl directory to the path so relative imports work when
# running from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ppo import PPOAgent
from environment.terraria_env import TerrariaEnv
from utils.checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.logger import TrainingLogger
from utils.normalizer import ObservationNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terraria RL Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Run mode: train or eval.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume from or evaluate.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints during training.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard and CSV logs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


# ---------------------------------------------------------------------------
# Seed setup
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config: Dict[str, Any],
    checkpoint_dir: str,
    log_dir: str,
    checkpoint_path_to_load: Optional[str],
) -> None:
    train_cfg = config["training"]
    ppo_cfg = config["ppo"]
    env_cfg = config["environment"]
    obs_cfg = config.get("observation", {})

    total_timesteps: int = train_cfg["total_timesteps"]
    checkpoint_interval: int = train_cfg["checkpoint_interval"]
    log_interval: int = train_cfg["log_interval"]
    rollout_length: int = ppo_cfg["rollout_length"]
    obs_normalization: bool = train_cfg.get("obs_normalization", True)

    frame_stack: int = obs_cfg.get("frame_stack", 1)
    base_obs_dim: int = env_cfg["observation_dim"]
    stacked_obs_dim: int = base_obs_dim * frame_stack

    device = torch.device(
        train_cfg.get("device", "cpu")
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    env = TerrariaEnv(config)
    agent = PPOAgent(config, stacked_obs_dim)
    normalizer = ObservationNormalizer(
        obs_dim=stacked_obs_dim,
        enabled=obs_normalization,
    )
    training_logger = TrainingLogger(log_dir=log_dir)

    update_count = 0
    global_step = 0

    # Optionally load checkpoint.
    if checkpoint_path_to_load:
        info = load_checkpoint(
            path=checkpoint_path_to_load,
            agent=agent,
            optimizer=agent.optimizer,
            normalizer=normalizer,
            device=device,
        )
        update_count = info["update_count"]
        global_step = info["total_timesteps"]
        logger.info(f"Resumed from checkpoint: update={update_count}, steps={global_step}")

    obs, _ = env.reset()
    obs = normalizer.update_and_normalize(obs)
    episode_reward = 0.0
    done = False

    logger.info(f"Starting training for {total_timesteps:,} timesteps")

    while global_step < total_timesteps:
        # ----- Collect rollout -----
        for step in range(rollout_length):
            actions, log_prob, value = agent.select_action(obs, deterministic=False)
            gym_actions = agent.action_space.actions_to_gym_dict(actions)

            next_obs, reward, terminated, truncated, info = env.step(gym_actions)
            next_obs_norm = normalizer.update_and_normalize(next_obs)

            episode_done = terminated or truncated
            agent.rollout_buffer.add(
                obs=obs,
                action=actions,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=episode_done,
            )

            episode_reward += reward
            global_step += 1
            obs = next_obs_norm

            if episode_done:
                ep_info = info
                training_logger.log_episode(
                    reward=ep_info.get("episode_reward", episode_reward),
                    length=ep_info.get("episode_length", step + 1),
                    win=ep_info.get("win", False),
                    boss_hp_remaining=ep_info.get("boss_hp_remaining", 1.0),
                    damage_dealt_ratio=ep_info.get("damage_dealt_ratio", 0.0),
                    damage_taken_ratio=ep_info.get("damage_taken_ratio", 0.0),
                    global_step=global_step,
                )
                episode_reward = 0.0
                obs, _ = env.reset()
                obs = normalizer.update_and_normalize(obs)

        # ----- Compute GAE -----
        episode_done_flag = terminated or truncated if 'terminated' in dir() else False
        agent.compute_gae(last_obs=obs, last_done=episode_done_flag)

        # ----- PPO update -----
        update_count += 1
        metrics = agent.train()

        if update_count % log_interval == 0:
            training_logger.log_update(
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                entropy=metrics["entropy"],
                approx_kl=metrics["approx_kl"],
                learning_rate=metrics["learning_rate"],
                global_step=global_step,
                action_entropies=metrics.get("action_entropies"),
            )
            logger.info(
                f"Update {update_count} | "
                f"step={global_step:,}/{total_timesteps:,} | "
                f"policy_loss={metrics['policy_loss']:.4f} | "
                f"value_loss={metrics['value_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"kl={metrics['approx_kl']:.4f} | "
                f"lr={metrics['learning_rate']:.2e}"
            )

        if update_count % checkpoint_interval == 0:
            ckpt = checkpoint_path(checkpoint_dir, update_count)
            save_checkpoint(
                path=ckpt,
                agent=agent,
                optimizer=agent.optimizer,
                normalizer=normalizer,
                update_count=update_count,
                total_timesteps=global_step,
                config=config,
            )

    # Final checkpoint.
    ckpt = checkpoint_path(checkpoint_dir, update_count)
    save_checkpoint(
        path=ckpt,
        agent=agent,
        optimizer=agent.optimizer,
        normalizer=normalizer,
        update_count=update_count,
        total_timesteps=global_step,
        config=config,
    )
    logger.info("Training complete.")
    training_logger.close()
    env.close()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    config: Dict[str, Any],
    checkpoint_path_to_load: str,
    log_dir: str,
    num_episodes: int = 10,
) -> None:
    env_cfg = config["environment"]
    obs_cfg = config.get("observation", {})
    train_cfg = config["training"]

    frame_stack: int = obs_cfg.get("frame_stack", 1)
    base_obs_dim: int = env_cfg["observation_dim"]
    stacked_obs_dim: int = base_obs_dim * frame_stack

    device = torch.device(
        train_cfg.get("device", "cpu")
        if torch.cuda.is_available()
        else "cpu"
    )

    env = TerrariaEnv(config)
    agent = PPOAgent(config, stacked_obs_dim)
    normalizer = ObservationNormalizer(
        obs_dim=stacked_obs_dim,
        enabled=train_cfg.get("obs_normalization", True),
    )
    training_logger = TrainingLogger(log_dir=os.path.join(log_dir, "eval"))

    load_checkpoint(
        path=checkpoint_path_to_load,
        agent=agent,
        optimizer=None,
        normalizer=normalizer,
        device=device,
    )
    agent.policy_network.eval()

    logger.info(f"Evaluating for {num_episodes} episodes (deterministic policy)")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = normalizer.normalize(obs)
        episode_reward = 0.0
        step = 0
        done = False

        while not done:
            actions, _, _ = agent.select_action(obs, deterministic=True)
            gym_actions = agent.action_space.actions_to_gym_dict(actions)
            obs, reward, terminated, truncated, info = env.step(gym_actions)
            obs = normalizer.normalize(obs)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        logger.info(
            f"Eval episode {ep + 1}/{num_episodes} | "
            f"reward={episode_reward:.2f} | "
            f"length={step} | "
            f"win={info.get('win', False)} | "
            f"boss_hp={info.get('boss_hp_remaining', 'N/A')}"
        )
        training_logger.log_episode(
            reward=episode_reward,
            length=step,
            win=info.get("win", False),
            boss_hp_remaining=info.get("boss_hp_remaining", 1.0),
            damage_dealt_ratio=info.get("damage_dealt_ratio", 0.0),
            damage_taken_ratio=info.get("damage_taken_ratio", 0.0),
            global_step=ep + 1,
        )

    training_logger.close()
    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seeds(seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "train":
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None and os.path.isdir(args.checkpoint_dir):
            ckpt_to_load = latest_checkpoint(args.checkpoint_dir)
            if ckpt_to_load:
                logger.info(f"Auto-resuming from latest checkpoint: {ckpt_to_load}")

        train(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            checkpoint_path_to_load=ckpt_to_load,
        )

    elif args.mode == "eval":
        if args.checkpoint is None:
            ckpt = latest_checkpoint(args.checkpoint_dir)
            if ckpt is None:
                logger.error("No checkpoint found. Specify --checkpoint or train first.")
                sys.exit(1)
            logger.info(f"Using latest checkpoint for eval: {ckpt}")
        else:
            ckpt = args.checkpoint

        evaluate(
            config=config,
            checkpoint_path_to_load=ckpt,
            log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()
