#!/usr/bin/env python
"""
Parallel PPO training across multiple Terraria instances.

Each headless tModLoader instance (see tml-instances.sh) runs BossMLMod on
its own TCP port (base_port + i). This script builds one SubprocVecEnv with
one TerrariaEnv per instance and trains a single PPO learner on all of them
simultaneously — N instances ≈ N× samples/second.

Usage (from the terraria_boss_agent directory, venv active):
    # 1. Start the game instances:
    #      ../tml-instances.sh start -n 4
    # 2. Train:
    python scripts/train_parallel.py --num-envs 4
    python scripts/train_parallel.py --num-envs 8 --base-port 8000 --new
    python scripts/train_parallel.py --num-envs 4 --config config/default.yaml \
        --set training.n_steps=2048 --set reward.boss_kill_bonus=800

Notes:
  * SB3's n_steps is PER ENV: rollout size = n_steps × num_envs. With many
    envs you usually want to lower training.n_steps accordingly.
  * batch_size should divide n_steps × num_envs.
  * Auto-resumes from the latest checkpoint like the stock trainer
    (use --new for a fresh model).
"""

import argparse
import copy
import glob
import logging
import os
import sys
from datetime import datetime

# Run from terraria_boss_agent/ or scripts/ — make `src` importable either way
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from src.env.terraria_env import TerrariaEnv

# Reuse the repo's own config loading, overrides, model construction,
# boss naming and win-rate logging so behaviour matches the stock trainer.
from src.training.train import (
    BOSS_NAMES,
    WinRateCallback,
    _build_model,
    apply_overrides,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_parallel")


def make_env_fn(config: dict, host: str, port: int, rank: int):
    """Closure creating one TerrariaEnv bound to a specific instance port."""

    def _init():
        cfg = copy.deepcopy(config)
        conn = dict(cfg.get("connection", {}))
        conn["host"] = host
        conn["port"] = port
        cfg["connection"] = conn
        env = TerrariaEnv(cfg)
        env._boss_type = cfg.get("boss_type", 0)
        return Monitor(env)

    return _init


def resolve_latest(model_dir: str) -> str | None:
    latest = os.path.join(model_dir, "latest.zip")
    if os.path.exists(latest):
        return latest
    ckpts = glob.glob(os.path.join(model_dir, "*.zip"))
    if ckpts:
        ckpts.sort(key=os.path.getmtime)
        return ckpts[-1]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Parallel PPO across N Terraria instances")
    p.add_argument("--num-envs", type=int, default=2,
                   help="Number of game instances / parallel envs (default: 2)")
    p.add_argument("--base-port", type=int, default=7777,
                   help="First instance port; env i connects to base+i (default: 7777)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--new", action="store_true", help="Start a fresh model")
    p.add_argument("--resume", default=None,
                   help="Checkpoint .zip to resume from (default: latest)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Config override, e.g. --set training.n_steps=2048")
    args = p.parse_args()

    config = load_config(args.config)
    config["_config_path"] = os.path.abspath(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    t_cfg = config.get("training", {})
    n = args.num_envs
    ports = [args.base_port + i for i in range(n)]
    logger.info(f"Parallel training: {n} envs on ports {ports}")

    n_steps = t_cfg.get("n_steps", 2048)
    batch_size = t_cfg.get("batch_size", 64)
    if (n_steps * n) % batch_size != 0:
        logger.warning(
            f"batch_size={batch_size} does not divide n_steps×num_envs={n_steps * n}; "
            f"SB3 will truncate minibatches. Consider --set training.batch_size=..."
        )

    # ── Vectorized env: one subprocess per game instance ─────────────────────
    env = SubprocVecEnv(
        [make_env_fn(config, args.host, port, i) for i, port in enumerate(ports)],
        start_method="fork",
    )
    frame_stack = t_cfg.get("frame_stack", 4)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
        logger.info(f"Frame stacking enabled: {frame_stack} frames")

    # ── Model dirs, mirroring the stock trainer's per-boss layout ────────────
    boss_id = config.get("boss_type", t_cfg.get("boss_type", 0))
    boss_name = BOSS_NAMES.get(boss_id, f"boss_{boss_id}") if boss_id else "unknown"
    log_dir = t_cfg.get("log_dir", "logs")
    model_dir = os.path.join(t_cfg.get("model_dir", "logs/models"), boss_name)
    os.makedirs(model_dir, exist_ok=True)

    resume_path = None if args.new else (args.resume or resolve_latest(model_dir))
    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env, tensorboard_log=log_dir)
    else:
        logger.info("Building fresh PPO model")
        model = _build_model(env, t_cfg, log_dir, use_lstm=t_cfg.get("use_lstm", False))

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        WinRateCallback(),
        CheckpointCallback(
            save_freq=max(t_cfg.get("checkpoint_freq", 100_000) // n, 1),
            save_path=model_dir,
            name_prefix=f"{boss_name}_par{n}_{run_tag}",
        ),
    ]

    total = t_cfg.get("total_timesteps", 2_000_000)
    logger.info(f"Training for {total:,} total timesteps across {n} envs "
                f"(rollout = {n_steps}×{n} = {n_steps * n} steps)")
    try:
        model.learn(
            total_timesteps=total,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=resume_path is None,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — saving model before exit.")
    finally:
        final = os.path.join(model_dir, f"{boss_name}_par{n}_{run_tag}_final.zip")
        model.save(final)
        latest = os.path.join(model_dir, "latest.zip")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(final), latest)
        except OSError:
            pass
        logger.info(f"Saved {final}")
        env.close()


if __name__ == "__main__":
    main()
