"""Configuration for RL fine-tuning. One dataclass, all defaults runnable as-is."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import os


def _devkit_root() -> Path:
    """RAP checkout root. The sbatch scripts export NAVSIM_DEVKIT_ROOT; fall back
    to this file's grandparent so interactive use needs no environment at all."""
    return Path(os.getenv("NAVSIM_DEVKIT_ROOT", Path(__file__).resolve().parents[1]))


@dataclass
class RLConfig:
    """Everything the RL pipeline needs. Field groups map 1:1 onto the modules."""

    # ------------------------------------------------------------------ data
    # Feature cache written by run_dataset_caching.py (agent.config.cache_data=True).
    # Same directory the supervised pipeline trains from -- RL reads the identical data.
    feature_cache_path: Path = field(default_factory=lambda: _devkit_root() / "cache" / "rap_ego")

    # PDM metric cache. Agent-independent: it stores the simulation/scoring state of the
    # scenario, and any trajectory is scored against it afterwards. Reused from DrivoR
    # rather than rebuilt (103288 tokens, a superset of the feature cache).
    # Anchored on the RAP checkout's parent, not $HOME: the sbatch scripts export
    # HOME=/fs/nexus-projects/sim2real/aliu but an interactive shell has the real
    # /nfshomes home, so Path.home() would resolve differently in the two contexts.
    metric_cache_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAP_METRIC_CACHE", _devkit_root().parent / "RAP" / "exp" / "train_metric_cache")
        )
    )

    # Where precompute.py writes the compact observations it distills from the two above.
    rl_cache_path: Path = field(default_factory=lambda: _devkit_root() / "cache" / "rl_obs")

    # Pretrained RAP weights. precompute.py runs this model once to produce both the
    # frozen scene latent and the base trajectory the policy writes residuals onto.
    checkpoint_path: Path = field(
        default_factory=lambda: _devkit_root() / "weights" / "RAP_DINO_navsimv2.ckpt"
    )

    # Fraction of tokens held out for evaluation. Split is deterministic (hash of token),
    # so train.py and eval.py agree without writing a split file.
    val_fraction: float = 0.05

    # ------------------------------------------------------------------- env
    # Trajectory horizon. RAP with time_horizon=5 / interval 0.5 emits 10 poses; the PDM
    # scorer consumes the first 8 (4 s). Read back from the cache, this is only a default.
    #
    # NOTE: poses 8 and 9 are dead weight for RL. compute_navsim_score.get_sub_score
    # truncates with `Trajectory(model_trajectory[:8])`, and rl/agent.py truncates the
    # same way at benchmark time, so 6 of the 30 action dimensions cannot change the
    # reward in either place. That is not a correctness bug -- both ends agree -- but the
    # policy still samples Gaussian noise into those dimensions, and their log-prob ratios
    # count towards approx_kl. With target_kl=0.05 as a backstop, dead dimensions make
    # updates get abandoned marginally earlier than they should. Shrinking the action to
    # the scored poses would fix it; it also changes what residual_scale means (the ramp
    # would reach full scale at pose 7 instead of 0.8x there), so it invalidates the
    # --probe calibration table above and any policy already trained. Left as-is
    # deliberately, documented rather than silently changed.
    num_poses: int = 10

    # Residual half-range at the FINAL pose, in (metres, metres, radians). The delta is
    # ramped linearly over the horizon so pose 0 barely moves and the trajectory stays
    # kinematically plausible -- a rigid shift of all poses would teleport the ego.
    #
    # Sized from `python rl/eval.py --probe`, measured over 40 navtrain scenes as the mean
    # PDMS cost of a fixed perturbation applied to RAP's own trajectory:
    #
    #     lateral 1 m   -0.01 (left) / -0.17 (right), collision rate 0.00
    #     lateral 4 m   -0.32 (left) / -0.52 (right), collision rate 0.25 / 0.30
    #     longitudinal x1.5  -0.36,  x0.5  -0.38
    #
    # So 1 m is the useful scale: large enough that the reward moves, small enough that
    # saturating the action cannot by itself cause a collision. At 4 m the policy can
    # destroy an already-good base trajectory faster than PPO can learn not to, and early
    # training is dominated by self-inflicted collisions.
    #
    # Raise these only if `--probe` shows a perturbation column that is flat against base.
    residual_scale: Tuple[float, float, float] = (1.0, 1.0, 0.1)

    # ---------------------------------------------------------------- reward
    # reward = w_pdm * pdm_score
    #        - w_collision * (1 - no_at_fault_collisions)
    #        - w_drivable  * (1 - drivable_area_compliance)
    #        - w_ttc       * (1 - time_to_collision)
    #
    # pdm_score already multiplies the collision and drivable terms in, so those two
    # weights are an *extra*, explicit penalty. They sharpen an otherwise flat signal:
    # PDMS differences between two safe trajectories are ~0.01, while a collision is a
    # cliff from ~0.85 to 0. Set them to 0 to optimise plain PDMS.
    w_pdm: float = 1.0
    w_collision: float = 1.0
    w_drivable: float = 0.5
    w_ttc: float = 0.25

    # Score the policy RELATIVE to RAP's own trajectory on the same scene:
    #     reward = r(policy_trajectory) - r(base_trajectory)
    #
    # This is not cosmetic. Most navtrain scenes are collision-free, so raw rewards inside
    # a PPO minibatch are nearly identical (~0.97 +/- 0.01). SB3 normalises advantages by
    # their minibatch std, and dividing near-identical returns by a near-zero std amplifies
    # pure noise into an enormous policy gradient -- measured approx_kl of 7e4 and
    # clip_fraction 0.92 on the very first updates. Centring on the per-scene base score
    # removes the shared component that carries no information about the action.
    #
    # It is also free: precompute.py scores each base trajectory once and stores the result,
    # so a rollout step still makes exactly one PDM call.
    relative_reward: bool = True

    # ------------------------------------------------------------------- ppo
    total_timesteps: int = 200_000
    n_envs: int = 16
    n_steps: int = 64          # per env; episodes are 1 step, so this is 64 scenes/env
    batch_size: int = 256
    learning_rate: float = 3e-4
    n_epochs: int = 10
    ent_coef: float = 0.0
    clip_range: float = 0.2
    # Episodes terminate after a single step, so there is nothing to bootstrap from and
    # the advantage is r - V(s) for any gamma. 0.0 says that outright.
    gamma: float = 0.0
    net_arch: Tuple[int, ...] = (512, 512)

    # Abandon an update once the policy has moved this far. A backstop, not the primary
    # fix -- relative_reward is that -- but a single pathological minibatch should not be
    # able to destroy a policy that starts from a pretrained planner.
    target_kl: float = 0.05
    # Normalise observations with a running mean/std. scene_latent is 5120 raw DINOv3
    # activations whose scale nothing else in the pipeline constrains.
    normalize_obs: bool = True

    seed: int = 0
    experiment_name: str = "rap_rl_ppo"

    @property
    def output_path(self) -> Path:
        """Run directory for checkpoints, tensorboard logs and eval output."""
        exp_root = Path(os.getenv("NAVSIM_EXP_ROOT", _devkit_root() / "exp"))
        return exp_root / self.experiment_name

    def validate(self, require_rl_cache: bool = True) -> None:
        """Fail early with an actionable message instead of deep inside a worker."""
        missing = []
        if require_rl_cache and not self.rl_cache_path.is_dir():
            missing.append(f"{self.rl_cache_path} -- run rl/precompute.py first")
        if not self.metric_cache_path.is_dir():
            missing.append(f"{self.metric_cache_path} -- set RAP_METRIC_CACHE")
        if missing:
            raise FileNotFoundError("Missing required cache(s):\n  " + "\n  ".join(missing))
