"""RAPPlanningEnv -- the NAVSIM PDM scorer wrapped as a gymnasium environment.

Episode structure
-----------------
One episode is one scene and exactly one step. That is not a simplification, it is
what the data supports: the metric cache is a frozen 4-second snapshot of a scenario
(observation, centerline, drivable area, other agents' futures). There is no way to
advance the world and re-observe, because the other agents are recorded, not
reactive. So the problem is a contextual bandit -- observe a scene, emit a whole
trajectory, receive its PDM score -- and every episode terminates after one step.
stable-baselines3 handles this without any special casing; see config.gamma.

Action
------
A bounded residual on RAP's own trajectory, not a trajectory from scratch. The base
already scores ~0.85 PDMS, so the policy starts on the useful part of the reward
surface instead of having to rediscover driving. The delta is ramped linearly over
the horizon (pose 0 nearly fixed, final pose gets the full residual_scale), which
keeps the trajectory anchored at the ego and kinematically plausible.

Observation
-----------
Dict of the four arrays precompute.py distilled. All frozen: this env runs no neural
network, it only indexes arrays and calls the PDM scorer.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import gymnasium as gym
import numpy as np

from rl.config import RLConfig
from rl.residual import apply_residual, delta_scale
from rl.reward import SCORE_KEYS, compute_reward, score_trajectory


class RLDataset:
    """The sharded output of precompute.py, loaded into RAM as flat arrays."""

    def __init__(self, rl_cache_path: Path):
        shards = sorted(Path(rl_cache_path).glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"No shard_*.npz in {rl_cache_path}. Run rl/precompute.py first."
            )

        tokens: List[str] = []
        buffers: Dict[str, List[np.ndarray]] = {}
        for shard in shards:
            with np.load(shard, allow_pickle=True) as data:
                tokens.extend(str(t) for t in data["tokens"])
                for key in ("scene_latent", "ego_status", "base_traj", "base_scores"):
                    buffers.setdefault(key, []).append(data[key])
                # base_pdm postdates the first version of the cache format. Fall back to
                # NaN so a stale cache fails loudly at reward time rather than silently
                # training against a zero baseline.
                buffers.setdefault("base_pdm", []).append(
                    data["base_pdm"] if "base_pdm" in data.files
                    else np.full((len(data["base_traj"]), len(SCORE_KEYS)), np.nan, np.float32)
                )

        self.tokens = tokens
        # scene_latent stays fp16 on disk but is promoted here: gymnasium Box spaces
        # are float32 and SB3 would cast on every single observation otherwise.
        self.scene_latent = np.concatenate(buffers["scene_latent"]).astype(np.float32)
        self.ego_status = np.concatenate(buffers["ego_status"]).astype(np.float32)
        self.base_traj = np.concatenate(buffers["base_traj"]).astype(np.float32)
        self.base_scores = np.concatenate(buffers["base_scores"]).astype(np.float32)
        self.base_pdm = np.concatenate(buffers["base_pdm"]).astype(np.float32)
        self.num_poses = self.base_traj.shape[1]

    def base_pdm_dict(self, index: int) -> Dict[str, float]:
        """True PDM sub-scores of the base trajectory, in SCORE_KEYS order."""
        return {key: float(self.base_pdm[index, i]) for i, key in enumerate(SCORE_KEYS)}

    def __len__(self) -> int:
        return len(self.tokens)


def split_tokens(tokens: List[str], val_fraction: float) -> Tuple[List[str], List[str]]:
    """Deterministic train/val split keyed on the token hash.

    Hashing rather than shuffling with a seed means train.py and eval.py agree on the
    split without either writing a file, and the split survives the cache being
    re-sharded or regenerated in a different order.
    """
    val, train = [], []
    threshold = val_fraction * 2 ** 32
    for token in tokens:
        digest = hashlib.md5(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little")
        (val if bucket < threshold else train).append(token)
    return train, val


# One MetricCacheLoader per process. Its index is a 103288-line CSV; re-reading it for
# every env in a SubprocVecEnv worker would be pure waste.
_METRIC_CACHE_PATHS: Dict[str, Dict[str, str]] = {}


def _metric_cache_paths(metric_cache_path: Path) -> Dict[str, str]:
    key = str(metric_cache_path)
    if key not in _METRIC_CACHE_PATHS:
        from navsim.common.dataloader import MetricCacheLoader

        _METRIC_CACHE_PATHS[key] = MetricCacheLoader(Path(metric_cache_path)).metric_cache_paths
    return _METRIC_CACHE_PATHS[key]


class RAPPlanningEnv(gym.Env):
    """Propose a trajectory for one cached NAVSIM scene, get its PDM score back."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        split: str = "train",
        dataset: Optional[RLDataset] = None,
    ):
        super().__init__()
        self.config = config or RLConfig()
        self.data = dataset or RLDataset(self.config.rl_cache_path)

        train_tokens, val_tokens = split_tokens(self.data.tokens, self.config.val_fraction)
        wanted = set(train_tokens if split == "train" else val_tokens)

        token_to_index = {t: i for i, t in enumerate(self.data.tokens)}
        available = _metric_cache_paths(self.config.metric_cache_path)
        # A token needs both an observation and a metric cache to be usable. On the
        # current setup all 14675 feature-cache tokens are present in the metric cache,
        # but the metric cache is a superset built for a different project, so this
        # guards against a partial regeneration rather than being dead code.
        # A token is usable only if it has an observation, a metric cache, AND a finite
        # baseline score. precompute.py writes NaN for scenes it could not score, so this
        # drops them rather than letting a NaN reward silently poison an update.
        has_baseline = np.isfinite(self.data.base_pdm).all(axis=1)
        self.indices = np.array(
            [token_to_index[t] for t in self.data.tokens
             if t in wanted and t in available and has_baseline[token_to_index[t]]],
            dtype=np.int64,
        )
        if len(self.indices) == 0:
            if not has_baseline.any():
                raise ValueError(
                    f"No token in {self.config.rl_cache_path} has a base_pdm baseline. "
                    "The observation cache predates the relative-reward format -- re-run "
                    "rl/precompute.py."
                )
            raise ValueError(f"No usable tokens for split={split!r}")
        dropped = len(wanted) - len(self.indices)
        if dropped:
            print(f"[env] split={split}: skipped {dropped} tokens with no metric cache "
                  f"or no baseline score")
        self.metric_cache_paths = available

        num_poses = self.data.num_poses
        latent_dim = int(np.prod(self.data.scene_latent.shape[1:]))
        ego_dim = int(np.prod(self.data.ego_status.shape[1:]))

        self.observation_space = gym.spaces.Dict({
            "scene_latent": gym.spaces.Box(-np.inf, np.inf, (latent_dim,), np.float32),
            "ego_status": gym.spaces.Box(-np.inf, np.inf, (ego_dim,), np.float32),
            "base_traj": gym.spaces.Box(-np.inf, np.inf, (num_poses * 3,), np.float32),
            "base_scores": gym.spaces.Box(0.0, 1.0, (6,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (num_poses * 3,), np.float32)

        # (P, 3) multiplier: residual_scale broadcast over xy/heading, ramped over time.
        # Shared with rl/agent.py rather than written twice -- if the env and the
        # benchmark agent ever applied different residuals, nothing would raise and
        # the trained policy would simply appear not to help. See rl/residual.py.
        self._delta_scale = delta_scale(num_poses, self.config.residual_scale)

        self._rng = np.random.default_rng(self.config.seed)
        self._cursor = 0
        self._index: int = int(self.indices[0])
        # Sequential order for eval, random for training: an eval pass must cover every
        # held-out scene exactly once for its mean PDMS to mean anything.
        self._sequential = split != "train"

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._sequential:
            self._index = int(self.indices[self._cursor % len(self.indices)])
            self._cursor += 1
        else:
            self._index = int(self._rng.choice(self.indices))

        return self._observation(), {"token": self.data.tokens[self._index]}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        trajectory = self._apply_residual(action)

        token = self.data.tokens[self._index]
        scores = score_trajectory(self.metric_cache_paths[token], trajectory)
        reward = compute_reward(scores, self.config)

        base_scores = self.data.base_pdm_dict(self._index)
        base_reward = compute_reward(base_scores, self.config)
        if self.config.relative_reward:
            # __init__ excludes tokens without a finite baseline, so this cannot fire
            # unless the cache changed under a running job.
            assert np.isfinite(base_reward), f"non-finite baseline for token {token}"
            reward -= base_reward

        # base_* keys let a callback log how often the policy actually beats RAP, which
        # the centred reward alone does not distinguish from "scene was easy".
        info = {
            "token": token,
            "trajectory": trajectory,
            "base_reward": base_reward,
            **scores,
            **{f"base_{k}": v for k, v in base_scores.items()},
        }
        # Episodes are one step by construction, so terminated is always True and
        # truncated always False -- there is no time limit to hit.
        return self._observation(), reward, True, False, info

    # -------------------------------------------------------------------- internals

    def _observation(self) -> Dict[str, np.ndarray]:
        i = self._index
        return {
            "scene_latent": self.data.scene_latent[i].reshape(-1),
            "ego_status": self.data.ego_status[i].reshape(-1),
            "base_traj": self.data.base_traj[i].reshape(-1),
            "base_scores": self.data.base_scores[i],
        }

    def _apply_residual(self, action: np.ndarray) -> np.ndarray:
        return apply_residual(self.data.base_traj[self._index], action, self._delta_scale)

    def base_trajectory(self) -> np.ndarray:
        """The unmodified RAP trajectory for the current scene -- the zero-action
        baseline eval.py compares against."""
        return self.data.base_traj[self._index].astype(np.float64)


def make_env(config: RLConfig, split: str, rank: int = 0, dataset=None):
    """Factory for SB3's DummyVecEnv/SubprocVecEnv, which want zero-arg callables."""

    def _init():
        env_config = RLConfig(**{**config.__dict__, "seed": config.seed + rank})
        env = RAPPlanningEnv(env_config, split=split, dataset=dataset)
        env.reset(seed=config.seed + rank)
        return env

    return _init
