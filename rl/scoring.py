"""True scores for a batch of candidate trajectories. Two backends, one layout.

Every row that ever enters the buffer is a `(6,)` vector in `SCORE_KEYS` order plus
a `(6,)` boolean mask saying which of those six columns actually carry a label. The
mask exists because the two sources of scenes are not equally observable:

  navtrain   full nuPlan metric cache -> the same `compute_navsim_score` the
             supervised pipeline trains against. All six columns are real.
  CARE clip  no map at all: `map_location` is the clip's own name and
             `roadblock_ids` is empty, so there is no drivable-area polygon and no
             route. Collision, TTC, progress and comfort are still measurable from
             the boxes and the human's own path; drivable_area_compliance is not.

Writing an unlabelled column as NaN and masking it in the loss is the only version
of this that stays honest. The alternative -- deriving drivability from the lane
polylines in the CARE export -- would produce a confident label from detections
that have no temporal association and a known over-extension bug, and a wrong
label is strictly worse than a missing one because nothing downstream can tell.

THREAD SAFETY: `compute_navsim_score` builds a module-level PDMSimulator/PDMScorer
pair and `PDMScorer` keeps per-call state on ``self`` (``_multi_metrics``,
``_ego_areas``, ...). Concurrent calls in one process interleave and corrupt each
other's results silently. Parallelise with processes, never threads.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

# Column order produced by compute_navsim_score.get_sub_score's np.stack. The CARE
# backend emits the identical layout so the two are interchangeable in the buffer.
SCORE_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision",
    "comfort",
    "pdm_score",
)
NUM_SCORES = len(SCORE_KEYS)
DRIVABLE_INDEX = SCORE_KEYS.index("drivable_area_compliance")
PDM_INDEX = SCORE_KEYS.index("pdm_score")

# The PDM scorer truncates every trajectory to its first 8 poses (4 s) --
# `get_sub_score` does `Trajectory(model_trajectory[:8])`. Both backends apply the
# same truncation so a trajectory is scored over the same horizon either way, and
# so poses beyond it cannot silently influence one backend and not the other.
SCORED_POSES = 8

# nuPlan ego footprint, from RAPConfig. The trajectory poses are at the rear axle,
# so the box centre sits `REAR_AXLE_TO_CENTER` ahead of the pose along its heading.
HALF_LENGTH = 2.588 + 0.25
HALF_WIDTH = 1.1485 + 0.1
REAR_AXLE_TO_CENTER = 1.461


# ---------------------------------------------------------------- navtrain backend


def score_navsim(metric_cache_path: str, trajectories: np.ndarray) -> np.ndarray:
    """True PDM sub-scores for K trajectories against one cached navtrain scenario.

    One call, not K calls. `get_sub_score` opens and lzma-decompresses the metric
    cache once and then simulates every proposal it was handed in a single batched
    `simulate_proposals`, so scoring 33 candidates costs barely more than scoring
    one. Calling it per trajectory would re-read a ~200 KB compressed pickle 33
    times per scene per round, which is most of the wall clock of a round.

    :param metric_cache_path: path to the scenario's metric_cache.pkl (lzma)
    :param trajectories: (K, P, 3) array of (x, y, heading) in the ego frame
    :return: (K, 6) float32, in SCORE_KEYS order; all-NaN rows if scoring raised
    """
    # Imported lazily: this pulls in nuplan's map stack, which costs seconds and is
    # pointless in a process that only ever loads CARE clips.
    from navsim.agents.rap_dino.score_module.compute_navsim_score import get_scores

    trajectories = np.asarray(trajectories, dtype=np.float64)
    assert trajectories.ndim == 3 and trajectories.shape[-1] == 3, (
        f"expected (K, P, 3), got {trajectories.shape}"
    )

    try:
        result = get_scores(
            [{"token": str(metric_cache_path), "poses": trajectories, "test": True}]
        )
        rows = np.asarray(result[0][0], dtype=np.float32)  # (K, 6)
    except Exception:
        # A malformed proposal can trip the simulator or shapely deep inside nuplan.
        # One bad scene must not take down a round that has already spent an hour
        # scoring; the row is written unlabelled and the buffer loader drops it.
        return np.full((len(trajectories), NUM_SCORES), np.nan, np.float32)

    if rows.shape != (len(trajectories), NUM_SCORES):
        raise ValueError(
            f"compute_navsim_score returned {rows.shape} for "
            f"{len(trajectories)} trajectories; expected "
            f"({len(trajectories)}, {NUM_SCORES})"
        )
    return rows


# -------------------------------------------------------------------- CARE backend


def ego_corners(poses: np.ndarray) -> np.ndarray:
    """Footprint corners of the ego box at each pose.

    :param poses: (..., 3) rear-axle (x, y, heading)
    :return: (..., 4, 2) corners, in the FRONT_LEFT, REAR_LEFT, REAR_RIGHT,
        FRONT_RIGHT order shapely's `polygons` wants (consistent winding; the
        actual starting corner is irrelevant to an intersects test).
    """
    poses = np.asarray(poses, dtype=np.float64)
    heading = poses[..., 2]
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    # Rear axle -> geometric centre.
    cx = poses[..., 0] + REAR_AXLE_TO_CENTER * cos_h
    cy = poses[..., 1] + REAR_AXLE_TO_CENTER * sin_h

    dx = np.array([HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH, HALF_LENGTH])
    dy = np.array([HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, -HALF_WIDTH])

    x = cx[..., None] + cos_h[..., None] * dx - sin_h[..., None] * dy
    y = cy[..., None] + sin_h[..., None] * dx + cos_h[..., None] * dy
    return np.stack([x, y], axis=-1)


def _ttc_projections(trajectories: np.ndarray) -> np.ndarray:
    """The trajectory, plus where it would be 0.5 s and 1.0 s later at constant speed.

    This is how the navsim scorer's time-to-collision term works and how the b2d
    scorer approximates it: rather than simulating further, each pose is pushed
    forward along its own instantaneous velocity and the same intersection test is
    run on the pushed-forward footprint. A trajectory that is not colliding but
    would be in a second scores 0 on TTC.

    :param trajectories: (K, P, 3)
    :return: (K, P, 3, 3) -- (candidate, pose, {t, t+0.5s, t+1s}, xyh)
    """
    xy = trajectories[..., :2]
    heading = trajectories[..., 2:]
    # Velocity of pose 0 is measured from the ego's own origin, which is where the
    # trajectory starts by construction (poses are in the current ego frame).
    previous = np.concatenate([np.zeros_like(xy[:, :1]), xy[:, :-1]], axis=1)
    velocity = (xy - previous) / 0.5

    return np.stack(
        [
            np.concatenate([xy, heading], axis=-1),
            np.concatenate([xy + velocity * 0.5, heading], axis=-1),
            np.concatenate([xy + velocity * 1.0, heading], axis=-1),
        ],
        axis=2,
    )


def _collision_masks(
    agent_corners: np.ndarray, candidate_corners: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate, per-timestep collision and TTC-collision flags.

    :param agent_corners: (N_agents, T, 4, 2) other agents' footprints over the
        horizon, in the current ego frame. An all-zero row means "this agent is not
        tracked at this timestep" and is skipped, matching the b2d scorer's
        `fut_mask`.
    :param candidate_corners: (K, T, 3, 4, 2) ego footprints, from `_ttc_projections`
    :return: (K, T) collided, (K, T) ttc_collided
    """
    import shapely
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    num_candidates, horizon = candidate_corners.shape[0], candidate_corners.shape[1]
    collided = np.zeros((num_candidates, horizon), dtype=bool)
    ttc_collided = np.zeros((num_candidates, horizon), dtype=bool)

    tracked = agent_corners.any(-1).any(-1)  # (N_agents, T)
    ego_polygons = shapely.creation.polygons(candidate_corners)

    for t in range(horizon):
        present = agent_corners[:, t][tracked[:, t]]
        if len(present) == 0:
            continue
        tree = STRtree([Polygon(box) for box in present], 10)

        hit = tree.query(ego_polygons[:, t, 0], predicate="intersects")
        collided[hit[0], t] = True

        # TTC uses the pushed-forward footprints only; index 0 is the real one and
        # is already covered by the collision test above.
        for projection in (1, 2):
            hit = tree.query(ego_polygons[:, t, projection], predicate="intersects")
            ttc_collided[hit[0], t] = True

    return collided, ttc_collided


def _progress(trajectories: np.ndarray, human: np.ndarray) -> np.ndarray:
    """Progress along the human's own path, as a ratio in [0, 1].

    With no map there is no centerline, so the human trajectory is the reference
    line -- which is exactly what the b2d scorer does. Each candidate's endpoint is
    projected onto that line and the result compared to the human's own progress,
    symmetrically: going twice as far scores the same as going half as far, because
    both are equally far from what the situation called for.

    A candidate that projects to the very end of the line has its distance from the
    human endpoint added, so overshooting is distinguishable from matching.
    """
    from shapely import Point
    from shapely.creation import linestrings

    line_points = np.concatenate([np.zeros((1, 2)), human[..., :2]])
    centerline = linestrings(line_points)
    human_progress = centerline.project(Point(line_points[-1]))

    raw = np.ones(len(trajectories))
    for index, candidate in enumerate(trajectories[..., :2]):
        projected = centerline.project(Point(candidate[-1]))
        if projected == human_progress:
            projected = projected + np.linalg.norm(candidate[-1] - human[-1, :2])
        raw[index] = projected
    raw = np.clip(raw, a_min=0.0, a_max=None)

    larger = np.maximum(raw, human_progress) + 0.01
    smaller = np.minimum(raw, human_progress) + 0.01
    return smaller / larger


def _comfort(trajectories: np.ndarray) -> np.ndarray:
    """Absolute-threshold approximation of nuPlan's comfort metric.

    nuPlan's version runs on the LQR-tracked 0.1 s states and checks six bounds.
    There is no simulator here, so this checks the two that survive differencing a
    0.5 s pose sequence -- acceleration magnitude and yaw rate -- at nuPlan's own
    thresholds.

    Deliberately absolute rather than relative to the human. The b2d scorer grades
    comfort against the ground-truth trajectory's own maximum, which works when the
    ground truth is a normal drive; on a CARE clip the human is *crashing*, so its
    peak deceleration is enormous and a relative bar would mark every candidate
    comfortable and carry no signal at all.
    """
    xy = trajectories[..., :2]
    previous = np.concatenate([np.zeros_like(xy[:, :1]), xy[:, :-1]], axis=1)
    velocity = (xy - previous) / 0.5
    acceleration = np.linalg.norm(np.diff(velocity, axis=1), axis=-1) / 0.5

    heading = np.unwrap(trajectories[..., 2], axis=-1)
    yaw_rate = np.abs(np.diff(heading, axis=-1)) / 0.5

    # nuPlan: max_abs_lon_accel 2.40, max_abs_lat_accel 4.89 -- 4.89 is the looser
    # of the two and this is a magnitude, not a decomposition. max_abs_yaw_rate 0.95.
    return ((acceleration <= 4.89).all(-1) & (yaw_rate <= 0.95).all(-1)).astype(np.float64)


def score_care(
    trajectories: np.ndarray,
    human_trajectory: np.ndarray,
    agent_corners: np.ndarray,
    label_drivable: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """True-ish sub-scores for K trajectories on one CARE clip window.

    The human trajectory is scored alongside the candidates and then *subtracted*:
    a candidate is only charged for a collision the human did not also have. On
    ordinary footage that changes nothing, and on a CARE clip it is the difference
    between a usable label and a constant. These clips are crashes -- the recorded
    ego hits something in most of them -- so without the subtraction every candidate
    that stays anywhere near the human's path scores zero on collision and the
    column tells the scorer nothing. Subtracting asks the question that matters
    instead: did this trajectory hit something the human avoided?

    :param trajectories: (K, P, 3) candidates in the current ego frame
    :param human_trajectory: (P, 3) the recorded ego future, same frame
    :param agent_corners: (N_agents, T, 4, 2) other agents over the horizon; T must
        be at least SCORED_POSES
    :param label_drivable: label drivable_area_compliance from lane detections.
        Off by default -- see the module docstring.
    :return: (K, 6) scores and a (6,) bool mask of which columns are labelled
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)[:, :SCORED_POSES]
    human_trajectory = np.asarray(human_trajectory, dtype=np.float64)[:SCORED_POSES]
    horizon = trajectories.shape[1]
    agent_corners = np.asarray(agent_corners, dtype=np.float64)[:, :horizon]

    # The human goes last, matching the b2d scorer's convention, so the collision
    # bookkeeping below can index it as [-1].
    everything = np.concatenate([trajectories, human_trajectory[None]], axis=0)

    collided, ttc_collided = _collision_masks(
        agent_corners, ego_corners(_ttc_projections(everything))
    )
    collided = collided[:-1] & ~collided[-1:]
    ttc_collided = ttc_collided[:-1] & ~ttc_collided[-1:]

    no_collision = 1.0 - collided.any(-1)
    time_to_collision = 1.0 - ttc_collided.any(-1)
    progress = _progress(trajectories, human_trajectory)
    comfort = _comfort(everything)[:-1]

    if label_drivable:
        raise NotImplementedError(
            "Drivable-area labels for CARE would have to come from the lane "
            "polylines in the export, which are per-frame detections with no "
            "temporal association. Build a real drivable-area source first."
        )
    # Unlabelled, so it must not change the aggregate either: the pdm_score column
    # is the product/weighted sum the navtrain scorer computes, and folding in a
    # placeholder drivable term would bias the one column inference actually ranks
    # on. Held at 1.0 inside the aggregate and masked out of the loss.
    drivable = np.ones_like(no_collision)

    # Same weights as compute_navsim_score's final score: the multiplicative gates
    # times a 5/12, 5/12, 2/12 blend of TTC, progress and comfort.
    multiplicative = no_collision * drivable
    pdm_score = multiplicative * (
        time_to_collision * 5.0 / 12.0 + progress * 5.0 / 12.0 + comfort * 2.0 / 12.0
    )

    scores = np.stack(
        [no_collision, drivable, progress, time_to_collision, comfort, pdm_score],
        axis=-1,
    ).astype(np.float32)

    mask = np.ones(NUM_SCORES, dtype=bool)
    mask[DRIVABLE_INDEX] = label_drivable
    return scores, mask


# ------------------------------------------------------------------ pool entry point


def score_job(job) -> Tuple[str, np.ndarray, np.ndarray]:
    """Score one scene's candidates. The entry point a worker pool maps over.

    It lives here, not in collect.py, on purpose. The pool uses the "spawn" start
    method, so every worker imports the module this function is defined in. Defined
    in collect.py it would pickle as ``__main__._score_one``, forcing each worker to
    re-import the whole collection script -- torch, the RAP model definitions, the
    DINOv3 weights path -- just to intersect some polygons.

    :param job: ``(token, source, trajectories, payload)`` where payload is the
        metric-cache path for "navtrain" and ``(human_trajectory, agent_corners,
        label_drivable)`` for "care"
    :return: (token, (K, 6) scores, (6,) label mask)
    """
    token, source, trajectories, payload = job
    if source == "navtrain":
        scores = score_navsim(payload, trajectories)
        return token, scores, np.ones(NUM_SCORES, dtype=bool)
    if source == "care":
        human_trajectory, agent_corners, label_drivable = payload
        scores, mask = score_care(
            trajectories, human_trajectory, agent_corners, label_drivable
        )
        return token, scores, mask
    raise ValueError(f"unknown source {source!r}")


def as_dict(row: Sequence[float]) -> Dict[str, float]:
    """One score row as a named dict, for logging."""
    return {key: float(row[i]) for i, key in enumerate(SCORE_KEYS)}
