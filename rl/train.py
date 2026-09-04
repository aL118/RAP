"""PPO fine-tuning of the RAP planner against the PDM score. Entry point.

Run order:  rl/precompute.py  ->  rl/train.py  ->  rl/eval.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RLConfig
from rl.env import RLDataset, make_env
from rl.reward import SCORE_KEYS


def build_vec_env(config: RLConfig, split: str, n_envs: int, use_subproc: bool = True):
    """Vectorised env.

    SubprocVecEnv, not threads: the PDM scorer keeps per-call state on a module-level
    singleton (see rl/reward.py), so concurrent scoring in one process corrupts results.
    Each worker re-reads the observation cache from disk (~300 MB) instead of inheriting
    it, because SB3 pickles the env factory through a pipe and a shared object would be
    serialised per worker anyway.
    """
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize,
    )

    env_fns = [make_env(config, split, rank) for rank in range(n_envs)]
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    # "spawn", not the Linux default "fork": the eval env is built after the PPO model,
    # by which point CUDA is initialised in the parent, and forking a CUDA context is
    # unsafe. spawn costs a few seconds of start-up and is correct in both places.
    kwargs = {"start_method": "spawn"} if vec_cls is SubprocVecEnv else {}
    env = VecMonitor(vec_cls(env_fns, **kwargs))

    if config.normalize_obs:
        # Observations only. The reward is already centred per scene by relative_reward,
        # and running-average reward normalisation on top would fight that and make
        # tensorboard's reward curve incomparable across runs.
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if split != "train":
            # EvalCallback copies the training env's statistics over before each
            # evaluation; leaving training=True here would let the eval env also drift
            # its own running mean between syncs, so the held-out numbers would be
            # produced under a normalisation the policy was never trained with.
            env.training = False
    return env


def _make_score_callback():
    """Callback logging PDM sub-scores to tensorboard.

    The episode reward alone hides which term moved. Collision rate in particular is
    the number this whole exercise is about, and it is invisible in a mean reward that
    is dominated by the progress term.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class PDMScoreCallback(BaseCallback):
        def __init__(self, log_every: int = 1000):
            super().__init__()
            self.log_every = log_every
            self._buffer: Dict[str, List[float]] = {}

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                for key in SCORE_KEYS + ("valid", "base_pdm_score"):
                    if key in info:
                        self._buffer.setdefault(key, []).append(float(info[key]))
                if "base_pdm_score" in info and "pdm_score" in info:
                    # The headline number: how often the policy beats the pretrained
                    # planner it is writing residuals onto.
                    self._buffer.setdefault("beat_base", []).append(
                        float(info["pdm_score"] > info["base_pdm_score"])
                    )

            if self._buffer and self.num_timesteps % self.log_every < self.training_env.num_envs:
                for key, values in self._buffer.items():
                    self.logger.record(f"pdm/{key}", float(np.mean(values)))
                # Explicit rate: "fraction of scenes with an at-fault collision".
                collisions = self._buffer.get("no_at_fault_collisions", [])
                if collisions:
                    self.logger.record("pdm/collision_rate", 1.0 - float(np.mean(collisions)))
                self._buffer.clear()
            return True

    return PDMScoreCallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rl-cache-path", type=Path, default=None,
                        help="Override the precomputed observation cache (smoke tests, ablations).")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Path to a .zip to continue from.")
    parser.add_argument(
        "--log-std-init", type=float, default=-2.0,
        help="Initial action log-std. The default std of 1.0 would make the very first "
             "residuals full-scale (metres), throwing away RAP's pretrained trajectory "
             "before PPO sees a single gradient. -2.0 is std~0.135, i.e. start near the base.",
    )
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Timesteps between held-out evaluations. 0 disables.")
    parser.add_argument("--no-subproc", action="store_true",
                        help="Single-process envs; slower, but tracebacks are readable.")
    args = parser.parse_args()

    config = RLConfig()
    for field in ("total_timesteps", "n_envs", "n_steps", "batch_size",
                  "learning_rate", "experiment_name", "seed",
                  "rl_cache_path", "val_fraction"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.validate()

    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[train] output -> {output_path}")

    # Load once up front purely to report the split; the workers load their own copies.
    dataset = RLDataset(config.rl_cache_path)
    print(f"[train] {len(dataset)} tokens, {dataset.num_poses} poses/trajectory")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    train_env = build_vec_env(config, "train", config.n_envs, use_subproc=not args.no_subproc)

    model_kwargs = dict(
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        target_kl=config.target_kl,
        seed=config.seed,
        device=args.device,
        tensorboard_log=str(output_path / "tb"),
        verbose=1,
        policy_kwargs=dict(
            net_arch=list(config.net_arch),
            log_std_init=args.log_std_init,
        ),
    )

    if args.resume:
        print(f"[train] resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=args.device)
    else:
        model = PPO("MultiInputPolicy", train_env, **model_kwargs)

    callbacks = [
        _make_score_callback()(),
        CheckpointCallback(
            save_freq=max(1, 20_000 // config.n_envs),
            save_path=str(output_path / "checkpoints"),
            name_prefix="ppo_rap",
        ),
    ]

    eval_env = None
    if args.eval_freq > 0:
        # Two envs is enough: the eval split is walked sequentially, so this only
        # controls how fast the fixed n_eval_episodes are scored.
        eval_env = build_vec_env(config, "val", 2, use_subproc=not args.no_subproc)
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=str(output_path / "best"),
            log_path=str(output_path / "eval"),
            eval_freq=max(1, args.eval_freq // config.n_envs),
            n_eval_episodes=64,
            deterministic=True,
        ))

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    finally:
        final_path = output_path / "final_model"
        model.save(str(final_path))
        # The running obs mean/std lives on the VecNormalize wrapper, not in the policy
        # zip. Without this file eval.py would feed the network unnormalised observations
        # and the trained policy would look broken.
        if config.normalize_obs:
            train_env.save(str(output_path / "vecnormalize.pkl"))
            print(f"[train] saved {output_path / 'vecnormalize.pkl'}")
        print(f"[train] saved {final_path}.zip")
        # Carry the observation cache's provenance next to the model. rl/agent.py reads
        # it to refuse a benchmark run that stacks this policy's residual on a different
        # RAP checkpoint than the one the observations were built from.
        meta_src = config.rl_cache_path / "meta.json"
        if meta_src.exists():
            (output_path / "obs_meta.json").write_text(meta_src.read_text())
            print(f"[train] saved {output_path / 'obs_meta.json'}")
        else:
            print(f"[train] WARNING: no {meta_src}; rl/agent.py cannot verify at eval "
                  f"time that the benchmark uses the checkpoint these observations "
                  f"came from. Re-run rl/precompute.py to generate it.")
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
