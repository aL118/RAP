"""Evaluate a trained policy, and probe how much signal the reward actually carries.

Two modes:

  (default)  Walk the held-out split twice -- once with the policy, once with a zero
             action -- and report both. The zero-action pass IS the pretrained RAP
             planner, so the delta between the two columns is the only number that
             says whether RL helped.

  --probe    No policy needed. Score a handful of hand-made trajectory perturbations
             per scene to show where the PDM reward has gradient and where it is flat.
             Run this before a long training job to sanity-check reward shaping.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RLConfig
from rl.env import RAPPlanningEnv
from rl.reward import SCORE_KEYS, score_trajectory


def _summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {key: float(np.mean([r[key] for r in rows])) for key in SCORE_KEYS}
    out["reward"] = float(np.mean([r["reward"] for r in rows]))
    out["collision_rate"] = 1.0 - out["no_at_fault_collisions"]
    return out


def _print_table(policy: Dict[str, float], base: Dict[str, float], n: int) -> None:
    print(f"\n{n} held-out scenes\n")
    print(f"{'metric':<28}{'RAP base':>12}{'RL policy':>12}{'delta':>12}")
    print("-" * 64)
    for key in list(SCORE_KEYS) + ["collision_rate", "reward"]:
        delta = policy[key] - base[key]
        # Lower is better for collision_rate; everything else is higher-is-better.
        marker = "" if abs(delta) < 1e-6 else (
            " *" if (delta < 0) == (key == "collision_rate") else "")
        print(f"{key:<28}{base[key]:>12.4f}{policy[key]:>12.4f}{delta:>+12.4f}{marker}")
    print("\n* = RL improved on the pretrained RAP planner")


def evaluate(model_path: Path, config: RLConfig, n_episodes: int, deterministic: bool) -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rl.env import RLDataset, make_env, split_tokens

    # Count from the token list rather than by constructing an env: RAPPlanningEnv would
    # load the whole observation cache, and the DummyVecEnv below loads it again anyway.
    _, val_tokens = split_tokens(RLDataset(config.rl_cache_path).tokens, config.val_fraction)
    n_episodes = min(n_episodes, len(val_tokens))
    print(f"[eval] {len(val_tokens)} val scenes available, evaluating {n_episodes}")

    # A 1-env DummyVecEnv, wrapped exactly as training wrapped it. The observation
    # statistics are part of the trained model: feeding raw observations to a policy
    # trained on normalised ones produces garbage that looks like a training failure.
    env = DummyVecEnv([make_env(config, "val", rank=0)])
    stats_path = Path(model_path).parent / "vecnormalize.pkl"
    if config.normalize_obs:
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_obs is on but {stats_path} is missing. It is written next to "
                "final_model.zip by train.py; evaluating without it is meaningless."
            )
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(model_path), device="auto")

    policy_rows, base_rows = [], []
    obs = env.reset()
    while len(policy_rows) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        for info, reward in zip(infos, rewards):
            policy_rows.append({**{k: info[k] for k in SCORE_KEYS}, "reward": float(reward)})
            base_rows.append({
                **{k: info[f"base_{k}"] for k in SCORE_KEYS},
                # The reward column must be comparable, so report the base's reward on the
                # same scale the policy is scored on: 0 when the reward is relative.
                "reward": 0.0 if config.relative_reward else info["base_reward"],
            })
        if len(policy_rows) % 50 < env.num_envs:
            print(f"[eval] {len(policy_rows)}/{n_episodes}")

    policy_rows, base_rows = policy_rows[:n_episodes], base_rows[:n_episodes]
    _print_table(_summarise(policy_rows), _summarise(base_rows), len(policy_rows))
    env.close()
    return 0


def probe(config: RLConfig, n_scenes: int) -> int:
    """Measure reward sensitivity to fixed trajectory perturbations.

    The point of this is calibration. On the 6 scenes checked while building this
    package, lateral offsets of up to 6 m changed nothing -- the PDM simulator tracks
    a proposal with an LQR controller that quietly absorbs them -- while doubling the
    longitudinal extent flipped no_at_fault_collisions to 0 in half of them. If a
    perturbation column here is identical to 'base', the policy gets no gradient from
    that direction and residual_scale needs raising.
    """
    env = RAPPlanningEnv(config, split="val")
    n_scenes = min(n_scenes, len(env.indices))

    def variants(base: np.ndarray) -> Dict[str, np.ndarray]:
        poses = len(base)
        ramp = np.linspace(0, 1, poses)
        out = {"base": base}
        for name, traj in (
            ("lon x1.5", base * np.array([1.5, 1.0, 1.0])),
            ("lon x0.5", base * np.array([0.5, 1.0, 1.0])),
        ):
            out[name] = traj
        for metres in (1.0, 4.0):
            for sign, side in ((1, "left"), (-1, "right")):
                traj = base.copy()
                traj[:, 1] += sign * metres * ramp
                out[f"{side} {metres:g}m"] = traj
        out["stop"] = np.zeros_like(base)
        return out

    totals: Dict[str, List[float]] = defaultdict(list)
    collisions: Dict[str, List[float]] = defaultdict(list)
    for _ in range(n_scenes):
        env.reset()
        path = env.metric_cache_paths[env.data.tokens[env._index]]
        for name, traj in variants(env.base_trajectory()).items():
            scores = score_trajectory(path, traj)
            totals[name].append(scores["pdm_score"])
            collisions[name].append(1.0 - scores["no_at_fault_collisions"])

    print(f"\nReward sensitivity over {n_scenes} scenes\n")
    print(f"{'perturbation':<16}{'mean PDMS':>12}{'vs base':>12}{'collision rate':>16}")
    print("-" * 56)
    base_mean = float(np.mean(totals["base"]))
    for name in totals:
        mean = float(np.mean(totals[name]))
        flat = " <- flat, no gradient" if name != "base" and abs(mean - base_mean) < 1e-4 else ""
        print(f"{name:<16}{mean:>12.4f}{mean - base_mean:>+12.4f}"
              f"{float(np.mean(collisions[name])):>16.4f}{flat}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to a PPO .zip. Defaults to <output_path>/final_model.zip")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using the distribution mean.")
    parser.add_argument("--probe", action="store_true",
                        help="Reward-sensitivity diagnostic; needs no trained model.")
    parser.add_argument("--probe-scenes", type=int, default=20)
    parser.add_argument("--rl-cache-path", type=Path, default=None,
                        help="Override the precomputed observation cache.")
    parser.add_argument("--val-fraction", type=float, default=None)
    args = parser.parse_args()

    config = RLConfig()
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.rl_cache_path is not None:
        config.rl_cache_path = args.rl_cache_path
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    config.validate()

    if args.probe:
        return probe(config, args.probe_scenes)

    model_path = args.model or (config.output_path / "final_model.zip")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model at {model_path}. Train one, or pass --probe.")
    return evaluate(Path(model_path), config, args.n_episodes, not args.stochastic)


if __name__ == "__main__":
    raise SystemExit(main())
