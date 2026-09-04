"""PDM sub-scores -> scalar reward.

Thin wrapper over the scorer the supervised pipeline already uses
(navsim.agents.rap_dino.score_module.compute_navsim_score), so RL and imitation
are scored by identical code.

THREAD SAFETY: compute_navsim_score builds a module-level PDMSimulator/PDMScorer
pair and PDMScorer keeps per-call state on ``self`` (``_multi_metrics``,
``_ego_areas``, ...). Concurrent calls in one process interleave and corrupt each
other's results. Parallelise with processes (SubprocVecEnv), never threads.
"""

from typing import Dict, Tuple
import numpy as np

# Column order produced by compute_navsim_score.get_sub_score's np.stack.
SCORE_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision",
    "comfort",
    "pdm_score",
)


def score_trajectory(metric_cache_path: str, poses: np.ndarray) -> Dict[str, float]:
    """Score one trajectory against one cached scenario.

    :param metric_cache_path: path to the scenario's metric_cache.pkl (lzma)
    :param poses: (num_poses, 3) array of (x, y, heading) in the ego frame.
        The scorer consumes the first 8 poses; extra poses are ignored, matching
        the supervised path exactly.
    :return: dict of SCORE_KEYS -> float, plus "valid" (0.0 if scoring raised).
    """
    # Imported lazily: this pulls in nuplan's map stack, which costs seconds and is
    # pointless in a process that only ever builds observations.
    from navsim.agents.rap_dino.score_module.compute_navsim_score import get_scores

    poses = np.asarray(poses, dtype=np.float64)
    assert poses.ndim == 2 and poses.shape[-1] == 3, f"expected (P, 3), got {poses.shape}"

    try:
        result = get_scores([{"token": metric_cache_path, "poses": poses[None], "test": True}])
        row = np.asarray(result[0][0])[0]  # (num_proposals=1, 6) -> (6,)
    except Exception:
        # A malformed proposal can trip the simulator or shapely deep inside nuplan.
        # Treat it as the worst possible outcome rather than killing the rollout: the
        # policy gets a strong negative signal and training continues.
        return {**{k: 0.0 for k in SCORE_KEYS}, "valid": 0.0}

    scores = {key: float(row[i]) for i, key in enumerate(SCORE_KEYS)}
    scores["valid"] = 1.0
    return scores


def score_row(job) -> np.ndarray:
    """Pool entry point: PDM sub-scores for one (metric_cache_path, poses) job.

    This lives here, not in precompute.py, on purpose. The scoring pool uses the "spawn"
    start method, so every worker imports the module this function is defined in. Defined
    in precompute.py it would be pickled as ``__main__._score_one`` -- forcing each worker
    to re-import the whole precompute script (torch, tqdm, the RAP model definitions) just
    to score a trajectory, and making the run depend on multiprocessing's __main__ fixup.
    Here it is ``rl.reward.score_row``, and a worker imports only numpy plus the scorer.

    :return: (len(SCORE_KEYS),) float32 row, in SCORE_KEYS order.
    """
    metric_cache_path, poses = job
    scores = score_trajectory(metric_cache_path, poses)
    return np.array([scores[k] for k in SCORE_KEYS], dtype=np.float32)


def compute_reward(scores: Dict[str, float], config) -> float:
    """Collapse PDM sub-scores into the scalar the agent optimises.

    pdm_score already folds the collision and drivable terms in multiplicatively, so
    the penalty terms are deliberate double-counting: they turn a soft difference into
    a sharp one. Two collision-free trajectories differ by ~0.01 PDMS, so without an
    explicit penalty the cliff at a collision is only visible through the same channel
    that carries the (much noisier) progress term.
    """
    reward = config.w_pdm * scores["pdm_score"]
    reward -= config.w_collision * (1.0 - scores["no_at_fault_collisions"])
    reward -= config.w_drivable * (1.0 - scores["drivable_area_compliance"])
    reward -= config.w_ttc * (1.0 - scores["time_to_collision"])
    return float(reward)
