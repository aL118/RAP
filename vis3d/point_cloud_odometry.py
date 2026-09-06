"""
Frame-to-frame ego motion by registering consecutive UniDepth point maps.

The alternative estimator to OpenVO (see estimate_ego_motion.py --method).
OpenVO's released checkpoints were trained on nuScenes with *ground-truth
LiDAR* depth -- weights/openvo's name, openvo_nusc_gt, says so, and its
training config points at depth_gt_intrs -- and on this footage they predict
motion roughly 20x too small: 0.022 m per frame pair on beepbeep against a
0.45 m median in the nuScenes ground truth they were fit to. That happens with
OpenVO's own Metric3D depth too, so it is the model being out of domain, not
the depth source.

This takes the opposite approach: no learned motion model at all. Metric scale
comes from the point maps themselves, which are already metric, so the estimate
cannot be off by a global factor the way a regressor's can. The cost is that it
is only as good as the depth and the feature tracks -- it has no prior to fall
back on when the scene is textureless.

Method, per consecutive pair:
  1. track corners with Lucas-Kanade, keeping only forward-backward consistent
     tracks (the standard guard against drift onto repeating texture),
  2. drop tracks on detected vehicles/pedestrians -- on a queue of traffic
     those dominate the image and would measure *their* motion, not ours,
  3. look the surviving tracks up in both frames' point maps to get 3D-3D
     correspondences,
  4. fit a rigid transform by RANSAC + Kabsch, refit on the inliers.
"""
import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("point_cloud_odometry needs opencv") from error

# Corner tracking. A few hundred well-spread corners is plenty for a rigid fit
# and keeps the per-pair cost negligible next to loading the point maps.
MAX_CORNERS = 1200
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 12
LK_WINDOW = (21, 21)
LK_LEVELS = 4

# A track is kept only if tracking it back to the first frame lands within this
# many pixels of where it started.
FORWARD_BACKWARD_TOLERANCE = 1.0

# Depth band kept for correspondences. Near points are where monocular depth is
# most reliable and where parallax is largest; beyond ~50 m the depth error
# swamps the frame-to-frame baseline, and those points would drag the fit.
MIN_DEPTH = 1.5
MAX_DEPTH = 50.0

# RANSAC over 3-point rigid fits. The threshold is the residual at which a
# correspondence counts as agreeing with the motion -- generous relative to
# depth noise, tight relative to a moving vehicle's displacement between frames.
RANSAC_ITERATIONS = 200
RANSAC_THRESHOLD = 0.25  # metres
MIN_CORRESPONDENCES = 12

# A rigid fit needs the points to span more than a line; a degenerate spread
# gives a rotation that is arbitrary about that axis.
MIN_POINT_SPREAD = 1.0  # metres, smallest singular value of the centred cloud


def _kabsch(source: np.ndarray, target: np.ndarray):
    """Rigid transform (R, t) taking `source` onto `target`, both (N, 3).

    Plain Kabsch with a reflection guard: the SVD of the covariance can come
    back with det = -1 (a mirror rather than a rotation) when the points are
    nearly coplanar, which a road surface very much is.
    """
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def _ransac_rigid(source: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    """Robust rigid fit, returning (R, t, inlier count) or None if it fails."""
    best_inliers = None
    for _ in range(RANSAC_ITERATIONS):
        sample = rng.choice(len(source), size=3, replace=False)
        try:
            rotation, translation = _kabsch(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm((source @ rotation.T + translation) - target, axis=1)
        inliers = residual <= RANSAC_THRESHOLD
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < MIN_CORRESPONDENCES:
        return None
    rotation, translation = _kabsch(source[best_inliers], target[best_inliers])
    return rotation, translation, int(best_inliers.sum())


def _track(previous_gray: np.ndarray, gray: np.ndarray, static_mask: np.ndarray):
    """Lucas-Kanade tracks from `previous_gray` to `gray`, restricted to
    `static_mask` and filtered by a forward-backward check. Returns two (N, 2)
    arrays of pixel coordinates, or None if too few survive."""
    corners = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=MAX_CORNERS, qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE, mask=static_mask.astype(np.uint8) * 255)
    if corners is None or len(corners) < MIN_CORRESPONDENCES:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, corners, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS,
        criteria=criteria)
    backward, _, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, forward, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS,
        criteria=criteria)

    round_trip = np.linalg.norm(corners - backward, axis=2).ravel()
    keep = (status.ravel() == 1) & (round_trip <= FORWARD_BACKWARD_TOLERANCE)
    if keep.sum() < MIN_CORRESPONDENCES:
        return None
    return corners[keep].reshape(-1, 2), forward[keep].reshape(-1, 2)


def _lookup(points_cam: np.ndarray, pixels: np.ndarray):
    """3D points at `pixels` from a (3, H, W) point map, plus a validity mask."""
    height, width = points_cam.shape[1:]
    columns = np.clip(np.round(pixels[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, height - 1)
    points = points_cam[:, rows, columns].T  # (N, 3)
    depth = points[:, 2]
    return points, (depth > MIN_DEPTH) & (depth < MAX_DEPTH) & np.isfinite(depth)


def relative_pose(previous_gray, gray, previous_points, points, static_mask):
    """The rigid transform from the previous frame's camera to this one's.

    Returns (4, 4) and the inlier count, or (None, 0) when the pair cannot be
    solved -- too few tracks, all of them on moving objects, or a degenerate
    spread. The caller decides what to do with a gap; holding the last pose is
    usually better than injecting a spurious jump.
    """
    tracked = _track(previous_gray, gray, static_mask)
    if tracked is None:
        return None, 0
    previous_pixels, pixels = tracked

    source, source_ok = _lookup(previous_points, previous_pixels)
    target, target_ok = _lookup(points, pixels)
    both = source_ok & target_ok
    if both.sum() < MIN_CORRESPONDENCES:
        return None, 0
    source, target = source[both], target[both]

    # Degenerate geometry check: the smallest singular value of the centred
    # cloud is how far it extends along its thinnest axis.
    spread = np.linalg.svd(source - source.mean(axis=0), compute_uv=False)
    if spread[-1] < MIN_POINT_SPREAD:
        return None, 0

    rng = np.random.default_rng(0)  # deterministic: same frames, same trajectory
    fit = _ransac_rigid(source, target, rng)
    if fit is None:
        return None, 0
    rotation, translation, inliers = fit

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, inliers


def estimate_trajectory(frame_paths, point_map_paths, static_masks=None, verbose=True):
    """Accumulates per-pair motion into camera-to-world poses.

    :param frame_paths: frame images, in temporal order
    :param point_map_paths: matching (3, H, W) UniDepth point maps
    :param static_masks: optional per-frame boolean masks, True where the pixel
        is static scene (see estimate_ego_motion._static_mask). None tracks
        everything, which is only safe when little of the view is moving.
    :return: (N, 4, 4) camera-to-world transforms starting at the identity, in
        OpenVO's convention so both estimators write the same pose file.
    """
    poses = [np.eye(4)]
    previous_gray = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    previous_points = np.load(point_map_paths[0])
    failures = 0

    for index in range(1, len(frame_paths)):
        gray = cv2.imread(str(frame_paths[index]), cv2.IMREAD_GRAYSCALE)
        points = np.load(point_map_paths[index])
        mask = static_masks[index - 1] if static_masks is not None \
            else np.ones(previous_gray.shape, bool)

        transform, inliers = relative_pose(previous_gray, gray, previous_points, points, mask)
        if transform is None:
            # Carry the previous pose forward: a failed pair means "unknown",
            # and repeating the last pose keeps the trajectory continuous
            # instead of teleporting the ego.
            failures += 1
            poses.append(poses[-1].copy())
        else:
            # transform maps previous-camera points into this camera; the
            # camera itself therefore moves by its inverse.
            poses.append(poses[-1] @ np.linalg.inv(transform))

        previous_gray, previous_points = gray, points

    if verbose and failures:
        print(f"{failures} of {len(frame_paths) - 1} pairs could not be solved "
              "(held the previous pose).")
    return np.stack(poses)
