"""
Main training script. Launches PPO (or RecurrentPPO) against the Terraria
BossMLMod TCP interface.

By default, training auto-resumes from the latest checkpoint for the
configured boss. Use --new to force a fresh model.

Usage:
    python -m src.training.train                      # auto-resume latest
    python -m src.training.train --new                # fresh model
    python -m src.training.train --resume path/to.zip # specific checkpoint
    python -m src.training.train --config path/to.yaml
"""

import argparse
import json
import os
import logging
import yaml
from collections import deque
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from tqdm.rich import tqdm
from stable_baselines3.common.monitor import Monitor

from src.env.terraria_env import TerrariaEnv
from src.network.policy import get_policy_kwargs
from src.training.curriculum import CurriculumCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Terraria boss NPC type IDs → short names
BOSS_NAMES = {
    4:   "eye_of_cthulhu",
    13:  "eater_of_worlds",
    35:  "skeletron",
    50:  "king_slime",
    113: "wall_of_flesh",
    125: "twins_ret",
    126: "twins_spaz",
    127: "skeletron_prime",
    134: "destroyer",
    222: "queen_bee",
    245: "golem",
    262: "plantera",
    370: "duke_fishron",
    398: "moon_lord",
    636: "empress_of_light",
    657: "queen_slime",
    668: "deerclops",
}

# ─────────────────────────────────────────────────────────────
#  Stabilization constants
# ─────────────────────────────────────────────────────────────
STAB_CLIP_FRACTION_HIGH    = 0.25
STAB_CLIP_FRACTION_LOW     = 0.05
STAB_KL_HIGH               = 0.03
STAB_LR_DECAY              = 0.75
STAB_LR_RECOVER            = 1.10
STAB_LR_MIN                = 1e-4
STAB_LR_MAX                = 5e-4
STAB_EV_COLLAPSE_THRESHOLD = 0.0
STAB_WINDOW                = 5
STAB_PLATEAU_WINDOW        = 50
STAB_PLATEAU_THRESHOLD     = 5.0
STAB_BEST_SAVE_FREQ        = 10


class ResumeProgressBarCallback(BaseCallback):
    """Progress bar that shows total steps including prior training."""

    def __init__(self, prior_steps: int = 0):
        super().__init__()
        self._prior_steps = prior_steps
        self.pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        total = self.locals["total_timesteps"]
        self.pbar = tqdm(total=total, initial=self._prior_steps)

    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.refresh()
        self.pbar.close()


class WinRateCallback(BaseCallback):
    """Tracks boss kill rate over a rolling window for logging."""

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.outcomes: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "boss_killed" in info:
                self.outcomes.append(info["boss_killed"])
                if len(self.outcomes) > self.window_size:
                    self.outcomes.pop(0)
                if len(self.outcomes) >= 10:
                    win_rate = sum(self.outcomes) / len(self.outcomes)
                    self.logger.record("rollout/win_rate", win_rate)
                if "episode_reward" in info:
                    self.logger.record("rollout/ep_reward", info["episode_reward"])
                if "episode_length" in info:
                    self.logger.record("rollout/ep_length", info["episode_length"])
        return True


class TrainingStabilizer(BaseCallback):
    """
    Automatic training stabilization.

    Watches clip_fraction, approx_kl, and explained_variance every iteration.
    Automatically adjusts learning rate to keep training stable.

    Actions taken:
      - clip_fraction > STAB_CLIP_FRACTION_HIGH  → reduce LR by decay factor
      - approx_kl > STAB_KL_HIGH                 → reduce LR by decay factor
      - explained_variance < 0                   → reduce LR (value fn collapsed)
      - all metrics healthy for STAB_WINDOW iters → small LR increase (recovery)
      - reward plateaus for STAB_PLATEAU_WINDOW   → log warning

    Also saves the best model by ep_rew_mean.
    """

    # Terminal color codes
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    GREEN  = "\033[32m"
    RESET  = "\033[0m"

    def __init__(
        self,
        model_dir:  str,
        run_prefix: str,
        initial_lr: float,
        verbose:    int = 1,
    ):
        super().__init__(verbose)
        self.model_dir  = model_dir
        self.run_prefix = run_prefix
        self.current_lr = initial_lr

        self._clip_history: deque[float] = deque(maxlen=STAB_WINDOW)
        self._kl_history:   deque[float] = deque(maxlen=STAB_WINDOW)
        self._ev_history:   deque[float] = deque(maxlen=STAB_WINDOW)
        self._rew_history:  deque[float] = deque(maxlen=STAB_PLATEAU_WINDOW)

        self._best_rew:        float     = float("-inf")
        self._best_model_path: str | None = None
        self._iter_count:      int       = 0

        self._reduce_cooldown:  int = 0
        self._recover_cooldown: int = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout (= once per iteration)."""
        self._iter_count += 1

        clip_frac = self._read("train/clip_fraction")
        kl        = self._read("train/approx_kl")
        ev        = self._read("train/explained_variance")
        ep_rew    = self._read("rollout/ep_rew_mean")

        # ── Always print stabilizer status ───────────────────────────
        clip_str = f"{clip_frac:.3f}" if clip_frac is not None else "n/a"
        kl_str   = f"{kl:.4f}"       if kl        is not None else "n/a"
        ev_str   = f"{ev:.3f}"       if ev        is not None else "n/a"
        rew_str  = f"{ep_rew:.1f}"   if ep_rew    is not None else "n/a"

        clip_colored = (
            f"{self.RED}{clip_str}{self.RESET}"
            if (clip_frac is not None and clip_frac > STAB_CLIP_FRACTION_HIGH)
            else f"{self.GREEN}{clip_str}{self.RESET}"
        )
        kl_colored = (
            f"{self.RED}{kl_str}{self.RESET}"
            if (kl is not None and kl > STAB_KL_HIGH)
            else f"{self.GREEN}{kl_str}{self.RESET}"
        )
        ev_colored = (
            f"{self.RED}{ev_str}{self.RESET}"
            if (ev is not None and ev < STAB_EV_COLLAPSE_THRESHOLD)
            else f"{self.GREEN}{ev_str}{self.RESET}"
        )

        print(
            f"  [Stabilizer] iter={self._iter_count:4d} | "
            f"lr={self.current_lr:.2e} | "
            f"clip={clip_colored} | "
            f"kl={kl_colored} | "
            f"ev={ev_colored} | "
            f"rew={rew_str}",
            flush=True,
        )

        if clip_frac is None or kl is None or ev is None:
            return  # not enough data yet

        self._clip_history.append(clip_frac)
        self._kl_history.append(kl)
        self._ev_history.append(ev)
        if ep_rew is not None:
            self._rew_history.append(ep_rew)

        # ── Decide whether to act ─────────────────────────────────────
        should_reduce  = False
        should_recover = False
        reason         = ""

        if self._reduce_cooldown > 0:
            self._reduce_cooldown -= 1
        if self._recover_cooldown > 0:
            self._recover_cooldown -= 1

        if clip_frac > STAB_CLIP_FRACTION_HIGH:
            should_reduce = True
            reason = f"clip_fraction={clip_frac:.3f} > {STAB_CLIP_FRACTION_HIGH}"

        if kl > STAB_KL_HIGH:
            should_reduce = True
            reason = f"approx_kl={kl:.4f} > {STAB_KL_HIGH}"

        if ev < STAB_EV_COLLAPSE_THRESHOLD:
            should_reduce = True
            reason = f"explained_variance={ev:.3f} collapsed"

        if (
            len(self._clip_history) == STAB_WINDOW
            and all(c < STAB_CLIP_FRACTION_LOW for c in self._clip_history)
            and all(k < STAB_KL_HIGH * 0.5     for k in self._kl_history)
            and all(e > 0.3                    for e in self._ev_history)
        ):
            should_recover = True

        # ── Apply LR change ───────────────────────────────────────────
        if should_reduce and self._reduce_cooldown == 0:
            new_lr = max(self.current_lr * STAB_LR_DECAY, STAB_LR_MIN)
            if new_lr != self.current_lr:
                print(
                    f"  {self.RED}[Stabilizer] ⚠  Reducing LR "
                    f"{self.current_lr:.2e} → {new_lr:.2e}  "
                    f"reason: {reason}{self.RESET}",
                    flush=True,
                )
                self._apply_lr(new_lr)
                self._reduce_cooldown  = STAB_WINDOW * 2
                self._recover_cooldown = STAB_WINDOW * 4

        elif should_recover and self._recover_cooldown == 0:
            new_lr = min(self.current_lr * STAB_LR_RECOVER, STAB_LR_MAX)
            if new_lr != self.current_lr:
                print(
                    f"  {self.GREEN}[Stabilizer] ✓  Healthy — recovering LR "
                    f"{self.current_lr:.2e} → {new_lr:.2e}{self.RESET}",
                    flush=True,
                )
                self._apply_lr(new_lr)
                self._recover_cooldown = STAB_WINDOW * 2

        # ── Plateau detection ─────────────────────────────────────────
        if len(self._rew_history) == STAB_PLATEAU_WINDOW:
            first_half  = list(self._rew_history)[:STAB_PLATEAU_WINDOW // 2]
            second_half = list(self._rew_history)[STAB_PLATEAU_WINDOW // 2:]
            improvement = (
                sum(second_half) / len(second_half)
                - sum(first_half) / len(first_half)
            )
            if abs(improvement) < STAB_PLATEAU_THRESHOLD:
                print(
                    f"  {self.YELLOW}[Stabilizer] ⚡ Plateau — "
                    f"improvement over last {STAB_PLATEAU_WINDOW} iters: "
                    f"{improvement:+.1f}. "
                    f"Consider changing preset or gear.{self.RESET}",
                    flush=True,
                )

        # ── Best model tracking ───────────────────────────────────────
        if (
            ep_rew is not None
            and ep_rew > self._best_rew
            and self._iter_count % STAB_BEST_SAVE_FREQ == 0
        ):
            self._best_rew = ep_rew
            path = os.path.join(self.model_dir, f"{self.run_prefix}_best")
            self.model.save(path)
            self._best_model_path = f"{path}.zip"
            print(
                f"  {self.GREEN}[Stabilizer] ★  New best model "
                f"(ep_rew_mean={ep_rew:.1f}) → {self._best_model_path}{self.RESET}",
                flush=True,
            )

        # ── Log current LR to tensorboard ────────────────────────────
        self.logger.record("train/current_lr", self.current_lr)

    def _read(self, key: str) -> float | None:
        """Read a value from the SB3 logger without crashing if missing."""
        try:
            val = self.logger.name_to_value.get(key)
            return float(val) if val is not None else None
        except Exception:
            return None

    def _apply_lr(self, new_lr: float) -> None:
        """Hot-patch the learning rate on the model's optimizer."""
        self.current_lr = new_lr
        self.model.learning_rate = new_lr
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr


class BossSwitchCallback(BaseCallback):
    """
    Stops training when the game reports a different boss_type than we started with.
    """

    def __init__(self, get_env_fn, initial_boss_type: int, verbose: int = 0):
        super().__init__(verbose)
        self._get_env          = get_env_fn
        self._initial_boss_type = initial_boss_type
        self.boss_changed      = False
        self.new_boss_type: int | None = None

    def _on_training_start(self) -> None:
        env = self._get_env()
        if env is None:
            logger.warning(
                "BossSwitchCallback: could not unwrap TerrariaEnv — "
                "boss switching is DISABLED."
            )
            return
        current = env._boss_type
        match   = current == self._initial_boss_type
        logger.info(
            f"BossSwitchCallback active: game_boss={current} "
            f"({BOSS_NAMES.get(current, f'boss_{current}')}), "
            f"training_boss={self._initial_boss_type} "
            f"({'OK' if match else 'MISMATCH — switching on first step'})"
        )

    def _on_step(self) -> bool:
        env = self._get_env()
        if env is None:
            return True
        current = env._boss_type
        if current > 0 and current != self._initial_boss_type:
            logger.info(
                f"Boss changed in game: {self._initial_boss_type} → {current} "
                f"({BOSS_NAMES.get(current, f'boss_{current}')}). "
                "Stopping training to switch models."
            )
            self.boss_changed  = True
            self.new_boss_type = current
            return False
        return True


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def make_env(config: dict) -> TerrariaEnv:
    return Monitor(TerrariaEnv(config))


def get_terraria_env(vec_env) -> TerrariaEnv | None:
    try:
        inner = vec_env
        while hasattr(inner, "venv") and not hasattr(inner, "envs"):
            inner = inner.venv
        env = inner.envs[0]
        while hasattr(env, "env"):
            env = env.env
        if isinstance(env, TerrariaEnv):
            return env
    except (AttributeError, IndexError):
        pass
    return None


def save_metadata(model_path: str, metadata: dict) -> None:
    meta_path = model_path.replace(".zip", "_meta.json")
    if not meta_path.endswith("_meta.json"):
        meta_path = model_path + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")


def load_metadata(model_path: str) -> dict | None:
    meta_path = model_path.replace(".zip", "_meta.json")
    if not meta_path.endswith("_meta.json"):
        meta_path = model_path + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def log_config(config: dict) -> None:
    r = config.get("reward",   {})
    t = config.get("training", {})
    e = config.get("episode",  {})

    logger.info("=" * 60)
    logger.info("ACTIVE CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Reward weights:")
    for k, v in sorted(r.items()):
        logger.info(f"  reward.{k} = {v}")
    logger.info("Training params:")
    for k in [
        "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "n_steps", "batch_size", "n_epochs", "total_timesteps",
    ]:
        logger.info(f"  training.{k} = {t.get(k, '(default)')}")
    logger.info("Episode:")
    for k, v in sorted(e.items()):
        logger.info(f"  episode.{k} = {v}")
    logger.info("=" * 60)


def _resolve_latest(model_dir: str, boss_name: str) -> str | None:
    latest_link = os.path.join(model_dir, "latest.zip")
    if os.path.exists(latest_link):
        return latest_link
    import glob
    checkpoints = glob.glob(os.path.join(model_dir, "*.zip"))
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime)
        path = checkpoints[-1]
        logger.info(
            f"No latest.zip symlink — using most recent checkpoint: "
            f"{os.path.basename(path)}"
        )
        return path
    logger.warning(f"No models found for {boss_name}, starting fresh")
    return None


def _build_model(env, t_cfg: dict, log_dir: str, use_lstm: bool):
    policy_kwargs = get_policy_kwargs(t_cfg)
    if use_lstm:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=t_cfg.get("learning_rate", 3e-4),
            gamma=t_cfg.get("gamma", 0.99),
            gae_lambda=t_cfg.get("gae_lambda", 0.95),
            clip_range=t_cfg.get("clip_range", 0.2),
            ent_coef=t_cfg.get("ent_coef", 0.01),
            vf_coef=t_cfg.get("vf_coef", 0.5),
            max_grad_norm=t_cfg.get("max_grad_norm", 0.5),
            n_steps=t_cfg.get("n_steps", 2048),
            batch_size=t_cfg.get("batch_size", 64),
            n_epochs=t_cfg.get("n_epochs", 10),
            policy_kwargs=dict(
                lstm_hidden_size=t_cfg.get("lstm_hidden_size", 256),
                **{k: v for k, v in policy_kwargs.items() if k != "net_arch"},
                net_arch=policy_kwargs.get(
                    "net_arch", dict(pi=[256, 256], vf=[256, 256])
                ),
            ),
            tensorboard_log=log_dir,
            verbose=1,
        )
        logger.info("Using RecurrentPPO (LSTM policy)")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=t_cfg.get("learning_rate", 3e-4),
            gamma=t_cfg.get("gamma", 0.99),
            gae_lambda=t_cfg.get("gae_lambda", 0.95),
            clip_range=t_cfg.get("clip_range", 0.2),
            ent_coef=t_cfg.get("ent_coef", 0.01),
            vf_coef=t_cfg.get("vf_coef", 0.5),
            max_grad_norm=t_cfg.get("max_grad_norm", 0.5),
            n_steps=t_cfg.get("n_steps", 2048),
            batch_size=t_cfg.get("batch_size", 64),
            n_epochs=t_cfg.get("n_epochs", 10),
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
        )
        logger.info("Using PPO (MLP policy)")
    return model


def _persist_boss_type(config: dict, boss_id: int) -> None:
    config_path = config.get("_config_path")
    if not config_path or not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw["boss_type"] = boss_id
        with open(config_path, "w") as f:
            yaml.dump(
                raw, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info(
            f"Config updated: boss_type = {boss_id} "
            f"({BOSS_NAMES.get(boss_id, f'boss_{boss_id}')})"
        )
    except Exception as exc:
        logger.warning(f"Could not update config file: {exc}")


def _update_latest_symlink(model_dir: str, filename: str) -> None:
    latest_link = os.path.join(model_dir, "latest.zip")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(filename, latest_link)
    except OSError:
        pass


def train(config: dict, resume_path: str | None = None) -> None:
    """
    Run PPO training, automatically switching models when the game boss changes.
    Includes automatic training stabilization via TrainingStabilizer callback.
    """
    log_config(config)

    t_cfg          = config.get("training", {})
    log_dir        = t_cfg.get("log_dir",   "logs")
    base_model_dir = t_cfg.get("model_dir", "logs/models")
    use_lstm       = t_cfg.get("use_lstm",  False)
    frame_stack    = t_cfg.get("frame_stack", 4)
    new_timesteps  = t_cfg.get("total_timesteps", 2_000_000)
    initial_lr     = float(t_cfg.get("learning_rate", 3e-4))

    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: make_env(config)])
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)
        logger.info(f"Frame stacking enabled: {frame_stack} frames")

    current_boss_id     = config.get("boss_type", t_cfg.get("boss_type", 0))
    current_resume_path = resume_path

    while True:
        boss_name = (
            BOSS_NAMES.get(current_boss_id, f"boss_{current_boss_id}")
            if current_boss_id else "unknown"
        )
        model_dir = os.path.join(base_model_dir, boss_name)
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Boss: {boss_name} (id={current_boss_id}) — models in {model_dir}")

        run_tag    = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = f"{boss_name}_{run_tag}"

        actual_resume = current_resume_path
        if actual_resume == "latest":
            actual_resume = _resolve_latest(model_dir, boss_name)

        resumed_metadata = None
        if actual_resume:
            logger.info(f"Resuming training from {actual_resume}")
            resumed_metadata = load_metadata(actual_resume)
            if resumed_metadata:
                logger.info(
                    f"Loaded metadata: boss_type={resumed_metadata.get('boss_type')}, "
                    f"has_loadout={resumed_metadata.get('loadout') is not None}"
                )

        terraria_env = get_terraria_env(env)
        if terraria_env is not None:
            terraria_env._boss_type = current_boss_id
            if resumed_metadata:
                terraria_env.set_loadout_on_connect(resumed_metadata.get("loadout"))

        if actual_resume:
            if use_lstm:
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(actual_resume, env=env)
            else:
                model = PPO.load(actual_resume, env=env)
            # Carry forward any LR the stabilizer set in a previous run
            if hasattr(model, "learning_rate"):
                resumed_lr = (
                    float(model.learning_rate)
                    if callable(model.learning_rate)
                    else float(model.learning_rate)
                )
                logger.info(f"Resumed model LR: {resumed_lr:.2e}")
                initial_lr = resumed_lr
        else:
            model = _build_model(env, t_cfg, log_dir, use_lstm)

        prior_steps = 0
        if actual_resume:
            import re
            m = re.search(r'_(\d+)_steps\.zip$', actual_resume)
            if m:
                prior_steps = int(m.group(1))
            if prior_steps == 0 and hasattr(model, "num_timesteps") and model.num_timesteps > 0:
                prior_steps = model.num_timesteps

        if prior_steps > 0:
            total_timesteps = prior_steps + new_timesteps
            logger.info(
                f"Resuming from {prior_steps:,} steps — "
                f"training {new_timesteps:,} more ({total_timesteps:,} total)"
            )
        else:
            total_timesteps = new_timesteps
            logger.info(f"Starting training for {new_timesteps:,} timesteps")

        # ── Build callbacks ───────────────────────────────────────────
        stabilizer = TrainingStabilizer(
            model_dir=model_dir,
            run_prefix=run_prefix,
            initial_lr=initial_lr,
        )

        boss_switch_cb = BossSwitchCallback(
            get_env_fn=lambda: get_terraria_env(env),
            initial_boss_type=current_boss_id,
        )

        callbacks = [
            CheckpointCallback(
                save_freq=t_cfg.get("checkpoint_interval", 50_000),
                save_path=model_dir,
                name_prefix=run_prefix,
            ),
            WinRateCallback(window_size=100),
            stabilizer,
            boss_switch_cb,
            ResumeProgressBarCallback(prior_steps=prior_steps),
        ]

        cur_cfg = config.get("curriculum", {})
        if cur_cfg.get("enabled", False):
            callbacks.append(CurriculumCallback(cur_cfg))
            logger.info("Curriculum learning enabled")

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,
        )

        # ── Save final model ──────────────────────────────────────────
        final_name = f"{run_prefix}_final_{total_timesteps // 1000}k"
        final_path = os.path.join(model_dir, final_name)
        model.save(final_path)
        terraria_env = get_terraria_env(env)
        if terraria_env is not None:
            save_metadata(f"{final_path}.zip", terraria_env.get_metadata())
        _update_latest_symlink(model_dir, f"{final_name}.zip")
        logger.info(f"Model saved: {final_path}.zip")

        if stabilizer._best_model_path:
            logger.info(
                f"Best model this run: {stabilizer._best_model_path} "
                f"(ep_rew_mean={stabilizer._best_rew:.1f})"
            )

        if not boss_switch_cb.boss_changed:
            logger.info("Training complete.")
            break

        current_boss_id     = boss_switch_cb.new_boss_type
        current_resume_path = "latest"
        new_boss_name       = BOSS_NAMES.get(current_boss_id, f"boss_{current_boss_id}")
        logger.info(f"Switching to boss: {new_boss_name} (id={current_boss_id})")
        _persist_boss_type(config, current_boss_id)
        initial_lr = float(t_cfg.get("learning_rate", 3e-4))  # reset LR on boss switch

    env.close()


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, val = item.split("=", 1)
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        if val.lower() in ("true", "false"):
            d[parts[-1]] = val.lower() == "true"
        else:
            try:
                d[parts[-1]] = int(val)
            except ValueError:
                try:
                    d[parts[-1]] = float(val)
                except ValueError:
                    d[parts[-1]] = val
        logger.info(f"Override: {key} = {d[parts[-1]]}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Terraria boss-fighting agent")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume", default=None, nargs="?", const="latest",
        help="Resume from checkpoint (default: latest for current boss).",
    )
    parser.add_argument(
        "--new", action="store_true",
        help="Force start a new model from scratch, ignoring any existing checkpoints",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override config value: --set reward.boss_damage_weight=2.0",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = os.path.abspath(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    resume_path = args.resume
    if args.new:
        resume_path = None
        logger.info("--new flag: starting fresh model from scratch")
    elif resume_path is None:
        resume_path = "latest"
        logger.info("Auto-resuming from latest checkpoint (use --new to start fresh)")

    train(config, resume_path=resume_path)


if __name__ == "__main__":
    main()