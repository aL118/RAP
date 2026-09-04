"""The action -> trajectory map, in one place.

This is deliberately its own module rather than a method on RAPPlanningEnv. Two
different processes have to apply the *identical* transform:

  * rl/env.py, during training, to score a proposed residual, and
  * rl/agent.py, at benchmark time, inside the NAVSIM devkit.

If those two ever disagree -- a different ramp, a transposed scale, a residual
applied in a different frame -- the policy still runs and still emits trajectories.
Nothing raises. The benchmark score just quietly reflects a different policy than
the one that was trained, and the run looks like RL simply failed to help. Sharing
the code is the only way to make that class of bug impossible rather than unlikely.

Importing this module must stay cheap: rl/agent.py runs inside the devkit's
evaluation stack, and rl/env.py pulls in gymnasium and the nuplan map stack. This
file imports numpy and nothing else.
"""

from typing import Sequence, Tuple

import numpy as np


def delta_scale(num_poses: int, residual_scale: Sequence[float]) -> np.ndarray:
    """Per-pose multiplier turning a [-1, 1] action into metres/radians.

    The residual is ramped linearly over the horizon: pose 0 gets 1/num_poses of
    ``residual_scale`` and the final pose gets all of it. A rigid shift applied
    equally to every pose would teleport the ego away from its current position at
    t=0, which is both kinematically impossible and something the PDM simulator's
    LQR tracker would partly absorb -- the reward would be measuring the tracker,
    not the plan.

    :param num_poses: trajectory length P
    :param residual_scale: half-range at the FINAL pose, as (x_m, y_m, heading_rad)
    :return: (P, 3) float32 multiplier
    """
    ramp = np.linspace(1.0 / num_poses, 1.0, num_poses, dtype=np.float32)
    return ramp[:, None] * np.asarray(residual_scale, dtype=np.float32)


def apply_residual(
    base_traj: np.ndarray, action: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Add a bounded, ramped residual to RAP's own trajectory.

    :param base_traj: (P, 3) RAP trajectory, the residual's anchor
    :param action: (P * 3,) or (P, 3) policy output, clipped to [-1, 1] here so
        callers cannot forget to -- a deterministic policy's mean is unbounded and
        SB3 does not clip it for you.
    :param scale: (P, 3) from :func:`delta_scale`
    :return: (P, 3) float64, the dtype the PDM scorer wants
    """
    base = np.asarray(base_traj, dtype=np.float64)
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(base.shape)
    return base + action * scale


def flat_action_dim(num_poses: int) -> int:
    """Length of the flattened action vector for a P-pose trajectory."""
    return num_poses * 3


def check_action_dim(policy_dim: int, num_poses: int) -> None:
    """Fail loudly when a policy is paired with a horizon it was not trained on.

    A 10-pose policy loaded against an 8-pose config would otherwise reshape a
    30-vector into (8, 3) and raise something opaque deep in numpy -- or worse,
    happen to fit and silently scramble the residual across poses.
    """
    expected = flat_action_dim(num_poses)
    if policy_dim != expected:
        raise ValueError(
            f"Policy emits {policy_dim} action dims but this config has "
            f"{num_poses} poses ({expected} dims). The policy was trained on a "
            f"{policy_dim // 3}-pose horizon -- set trajectory_sampling.time_horizon "
            f"to the value rl/precompute.py ran with."
        )
