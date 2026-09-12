"""
Metric ego motion from the road plane, with no depth network in the loop.

Why this exists: the pipeline's other two estimators both take their scale from
UniDepth point maps, and on 18 of the 50 CARE clips those maps are unusable --
the ego's own hood comes back at 25-30 m, the scene flattens into a slab, and
the resulting trajectories are 20-100x short with the direction essentially
random (changelane travels a GPS-measured 265 m and is logged as 2.6 m net).
Metric3D fails on the same clips, so swapping depth models does not help. What
does help is not needing depth at all.

The anchor here is the road plane, but the pose and the scale are recovered
separately, because they want different evidence:

  pose   the essential matrix over every static feature in the frame, which
         gives rotation and the *direction* of translation robustly and needs
         no depth at all. This is the mature, well-conditioned half.
  scale  triangulate the road-surface features with that pose, fit a plane to
         them, and stretch the whole estimate until that plane sits the known
         camera height below the camera.

Decoupling them is not a preference, it is forced. The obvious approach --
one homography of the road plane, decomposed into rotation, normal and
translation-over-plane-distance -- was built first and does not work here: at
11.5 m/s and 10 Hz the ego moves ~1.15 m per frame against a ~1.3 m camera
height, so |t|/d is near 1, the near road is entirely replaced between frames,
and the only road left in both views is far enough away that its homography is
degenerate with a pure rotation. Measured on blocker it solved about half the
pairs, returned 0.15-2.4 m/s against a true 11.5, and flipped the sign of
forward motion depending on which rows were tracked.

Deliberately not lane markings. Dash-pitch counting works (validated against
changelane's burned-in GPS, which puts its markings at a 16.3 m cycle) but only
where there are markings to count, and it inherits whatever the local marking
standard is. Road surface is present in every dashcam clip; paint is not. Where
lanes *do* exist, camera_height_from_lanes() uses their width to measure the one
free parameter this estimator has, rather than to drive it.

Scale caveat worth stating plainly: a wrong camera height is a single global
factor on the whole trajectory -- 1.2 m assumed against a true 1.5 m is 20%
slow, uniformly. That is a different class of error from what it replaces, where
the magnitude was out by 100x and the sign of forward motion flipped on 9 clips.

STATUS, 2026-09-12: the pose half works, the scale half does not yet.

Three bugs that stopped it solving a single pair are fixed (see relative_pose
and ROAD_BAND). What remains is not a bug but a measurement, and it is the same
wall the homography design hit: at 21 m/s and 10 Hz the near road is not
trackable frame to frame. Measured on changelane frame 0->1, whose true speed
its burned-in OSD gives as 76 km/h:

  row 620   true flow    8 px     LK 2.8    DIS 3.1    exhaustive search 4.0
  row 730   true flow   48 px     LK 11.5   DIS 6.9    exhaustive search 8.0
  row 800   true flow  107 px     LK 4.4    DIS 5.1    no confident match
  row 900   true flow  244 px     LK 0.0    DIS 0.0    no confident match

The true figures are geometry, and they are confirmed by eye: a lane marking
crosses row 690, then 730, then 770 on three consecutive frames. All three
matchers agree with each other and disagree with the truth, which is what
self-similar asphalt under a 2x per-frame scale change does to any local
matcher -- they lock onto a false nearby match rather than fail loudly.

So triangulating road features across consecutive frames recovers a plane
16x too far away, and the metre-per-frame it implies is correspondingly small.
Do not trust this module's scale at highway speed until that is solved. The
pose it returns -- rotation and the direction of translation -- is sound, and
is the part worth reusing.

What was tried against that and rejected: an inverse-perspective (bird's-eye)
warp of the road plus 2-D phase correlation, which locks onto the lane lines and
tyre-polish streaks running *parallel* to the motion, and so carries no forward
information; and the same warp reduced to a 1-D longitudinal profile, which is
right on some pairs (+1.44 m against a true 2.11) and near zero on others.
Choosing the warp's pitch by maximising the correlation peak is a trap: it
selects -6 deg, where the patch degenerates to a nearly constant profile that
correlates at 0.99 with itself at zero shift, and reports a stationary car.
The warp geometry itself is right -- the dashes visibly slide about 2 m per
frame -- and calibrate_ground.py already supplies its pitch, yaw and roll from
the lane vanishing point, so a working estimator on top of it is still the most
promising route.
"""
import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("ground_plane_odometry needs opencv") from error

from point_cloud_odometry import _track, MIN_CORRESPONDENCES

# Windscreen-mounted dashcams sit roughly level with the driver's eyeline. This
# is the estimator's only free parameter when a clip has no lane markings to
# measure it from; see camera_height_from_lanes.
DEFAULT_CAMERA_HEIGHT = 1.3

# Standard through-lane width. Ontario/TAC and US MUTCD freeways are both 3.6-3.7 m,
# and European motorways 3.5-3.75 m, so this is far better constrained across
# jurisdictions than dash pitch (which ranges 12-16 m between the clips here).
DEFAULT_LANE_WIDTH = 3.65

# Essential-matrix RANSAC, in pixels on a 1920-wide frame: generous enough for
# LK jitter on wet asphalt, tight enough to reject a vehicle under its own motion.
EPIPOLAR_THRESHOLD = 1.5
MIN_POSE_CORRESPONDENCES = 40

# Passed to recoverPose so its cheirality test keeps distant points instead of
# silently dropping everything past ~50 baselines. See relative_pose.
RECOVER_POSE_DISTANCE = 1e9

# Plane fit over the triangulated road points.
MIN_PLANE_POINTS = 30
PLANE_INLIER_FRACTION = 0.10   # of the plane distance, so it scales with the fit
PLANE_ITERATIONS = 200

# Triangulated points near the epipole have almost no parallax under forward
# motion and their depth is meaningless; require this much image displacement.
MIN_PARALLAX_PX = 3.0

# The plane normal a road must have, in camera coordinates (x right, y down,
# z forward). It is +y, pointing *down* into the road, not up: OpenCV returns
# the plane as n.X = d with d > 0, and the camera at the origin gives n.0 = 0 < d,
# so n has to point from the camera towards the plane. Getting this backwards
# selects the mirrored twin of the real solution, which reports forward motion
# with the wrong sign. Decomposing a homography yields up to four (R, t, n) and
# this is what picks the physical one; loose enough to allow pitch and roll.
NORMAL_DOWN_TOLERANCE = 0.75

# Asphalt is far less corner-rich than the general scene, and the default
# quality level -- relative to the strongest corner found -- keeps too little of
# it. Road tracking needs the weaker corners that texture, tar seams and surface
# wear provide.
ROAD_CORNER_QUALITY = 0.003

# Rows of the frame to track in, as fractions of height. The top excludes sky
# and the horizon, where road points are too far to carry usable parallax; the
# bottom excludes the bonnet and the wiper's parked position.
#
# 0.93 did NOT clear the bonnet. On changelane the bonnet fills rows 840-1010,
# so the old band put 463 of 815 road corners on the car's own bodywork --
# perfectly trackable, perfectly static, and 57% of the corner budget spent on
# a surface that cannot measure anything. They are dropped later by
# MIN_PARALLAX_PX, so the symptom was starvation rather than a wrong plane.
# Prefer a drivable mask where the clip has one; this is the fallback.
ROAD_BAND = (0.55, 0.78)


def road_mask(shape, drivable=None, static=None):
    """Where to look for road-plane features.

    `drivable` is the pipeline's own drivable_masks/ if the clip has them. When
    it does not -- which is the case this module is built to survive -- the band
    of rows in front of the car is a serviceable stand-in, because a forward
    dashcam view is road there by construction. `static` is the usual
    not-a-vehicle mask; road paint and texture under a car in front would
    otherwise measure that car's motion.
    """
    height, width = shape
    mask = np.zeros(shape, bool)
    mask[int(height * ROAD_BAND[0]):int(height * ROAD_BAND[1])] = True
    if drivable is not None:
        drivable = drivable.astype(bool)
        # Only trust the drivable mask when it actually covers something; an
        # empty or near-empty one means the detector failed, not that there is
        # no road, and intersecting with it would leave nothing to track.
        if drivable.sum() > 0.02 * height * width:
            mask &= drivable
    if static is not None:
        mask &= static.astype(bool)
    return mask


def _select_decomposition(rotations, translations, normals):
    """The physically possible (R, t/d, n) out of the up-to-four OpenCV returns.

    Two filters do it: the plane has to be below the camera with its normal
    pointing up, and the motion has to be small -- consecutive frames at 10 Hz
    are a tenth of a second apart, so a solution implying a large rotation is
    the mirrored twin rather than the real one.
    """
    best = None
    for rotation, translation, normal in zip(rotations, translations, normals):
        normal = normal.ravel()
        if normal[1] < NORMAL_DOWN_TOLERANCE:     # y is down; see the constant
            continue
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        if best is None or angle < best[0]:
            best = (angle, rotation, translation.ravel(), normal)
    return None if best is None else best[1:]


def _fit_plane(points, rng):
    """RANSAC plane through triangulated road points -> (unit normal, distance).

    Returned so that the normal points *down* into the road (+y in camera
    coordinates) and the distance is positive, matching the sign convention the
    scale step expects.
    """
    best = None
    for _ in range(PLANE_ITERATIONS):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        distance = normal @ sample[0]
        if distance < 0:
            normal, distance = -normal, -distance
        residual = np.abs(points @ normal - distance)
        inliers = residual <= PLANE_INLIER_FRACTION * max(distance, 1e-6)
        if best is None or inliers.sum() > best[0].sum():
            best = (inliers, normal, distance)
    if best is None or best[0].sum() < MIN_PLANE_POINTS:
        return None
    # Refit on the inliers: the three-point sample fixes which plane, the refit
    # fixes where it is.
    inliers = best[0]
    centre = points[inliers].mean(axis=0)
    _, _, vt = np.linalg.svd(points[inliers] - centre)
    normal = vt[-1]
    distance = normal @ centre
    if distance < 0:
        normal, distance = -normal, -distance
    if normal[1] < NORMAL_DOWN_TOLERANCE:
        return None
    return normal, float(distance), int(inliers.sum())


# Road features are tracked separately from the pose features, so they were
# never scored by findEssentialMat's RANSAC. This re-scores them against the
# recovered epipolar geometry, in pixels, before they are triangulated.
SAMPSON_THRESHOLD = 2.0


def _sampson(essential, previous_pixels, pixels, intrinsics):
    """Sampson distance of each correspondence to the epipolar geometry, in px."""
    inverse = np.linalg.inv(intrinsics)
    first = inverse @ np.hstack([previous_pixels, np.ones((len(previous_pixels), 1))]).T
    second = inverse @ np.hstack([pixels, np.ones((len(pixels), 1))]).T
    e_first, e_second = essential @ first, essential.T @ second
    numerator = np.sum(second * e_first, axis=0) ** 2
    denominator = e_first[0] ** 2 + e_first[1] ** 2 + e_second[0] ** 2 + e_second[1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12)) * intrinsics[0, 0]


def relative_pose(previous_gray, gray, static_mask, plane_mask, intrinsics, height):
    """Motion from the previous camera to this one, in metres.

    Returns ((4, 4), inlier count, plane normal), or (None, 0, None) when the
    pair will not solve. The transform maps previous-camera points into this
    camera, matching point_cloud_odometry.relative_pose so estimate_trajectory
    composes them the same way.
    """
    # Pose pass: the whole static scene at the module's usual corner quality.
    # This is deliberately a separate call from the road pass below. Selecting
    # road points out of one whole-frame track does not work -- goodFeaturesToTrack
    # scores corners relative to the strongest in the *frame*, and asphalt never
    # competes with poles, barriers and vehicles: on changelane only 6-9% of the
    # tracked points landed on the road (31-65 of ~550), and after intersecting
    # with the pose inliers essentially none survived, so every pair failed.
    # Tracking the road region on its own yields 489-815 points there instead.
    tracked = _track(previous_gray, gray, static_mask)
    if tracked is None:
        return None, 0, None
    previous_pixels, pixels = tracked
    if len(previous_pixels) < MIN_POSE_CORRESPONDENCES:
        return None, 0, None

    essential, mask = cv2.findEssentialMat(
        previous_pixels, pixels, intrinsics, method=cv2.RANSAC,
        prob=0.999, threshold=EPIPOLAR_THRESHOLD)
    if essential is None or essential.shape != (3, 3):
        return None, 0, None
    # Gate on the epipolar inliers, not on recoverPose's count. recoverPose
    # defaults to distanceThresh=50, which in units where |t| == 1 discards every
    # point beyond ~50 baselines -- about 57 m at highway speed, i.e. most of a
    # freeway scene. It changes nothing about R and t (measured identical from 50
    # to 1e9) and only shrinks the returned count, so gating on that count
    # rejected healthy pairs: 597 epipolar inliers came back as a count of 33.
    if int(mask.sum()) < MIN_POSE_CORRESPONDENCES:
        return None, 0, None
    _, rotation, unit_translation, _, _ = cv2.recoverPose(
        essential, previous_pixels, pixels, intrinsics, mask=mask.copy(),
        distanceThresh=RECOVER_POSE_DISTANCE)
    unit_translation = unit_translation.ravel()

    # Scale pass: track the road region on its own, keeping weaker corners, then
    # triangulate those under the pose above and see how far the road plane sits
    # below the camera in these (unit-translation) units.
    road = _track(previous_gray, gray, plane_mask,
                  quality=ROAD_CORNER_QUALITY, min_distance=8)
    if road is None:
        return None, 0, None
    road_previous, road_pixels = road
    parallax = np.linalg.norm(road_pixels - road_previous, axis=1) >= MIN_PARALLAX_PX
    epipolar = _sampson(essential, road_previous, road_pixels, intrinsics) <= SAMPSON_THRESHOLD
    keep = parallax & epipolar
    if keep.sum() < MIN_PLANE_POINTS:
        return None, 0, None

    projection0 = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection1 = intrinsics @ np.hstack([rotation, unit_translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(projection0, projection1,
                                        road_previous[keep].T, road_pixels[keep].T)
    points = (homogeneous[:3] / homogeneous[3]).T
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    if len(points) < MIN_PLANE_POINTS:
        return None, 0, None

    fit = _fit_plane(points, np.random.default_rng(0))
    if fit is None:
        return None, 0, None
    normal, distance, plane_inliers = fit

    # The plane is `distance` away in units where |translation| == 1; it is
    # `height` away in metres. That ratio is the whole scale recovery, and it is
    # the only place metres enter.
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = unit_translation * (height / distance)
    return transform, plane_inliers, normal


def camera_height_from_lanes(lane_mask, intrinsics, normal,
                             lane_width=DEFAULT_LANE_WIDTH, row_fractions=(0.80, 0.88)):
    """Camera height in metres from the width of a lane, or None.

    Back-projects the lane edges onto the plane with unit distance, so the
    separation that comes back is in units of the camera height; the true width
    then fixes it. Only used to *measure* what this module otherwise assumes,
    and only when a clip has two clean lane edges to measure between.
    """
    height_px, width_px = lane_mask.shape
    inverse = np.linalg.inv(intrinsics)
    widths = []
    for fraction in row_fractions:
        row = int(height_px * fraction)
        columns = np.flatnonzero(lane_mask[row] > 0)
        if len(columns) < 2:
            continue
        # Outermost two marks on the row: the ego lane's own edges, provided the
        # camera sits between them -- true for a forward dashcam in its lane.
        left, right = columns.min(), columns.max()
        if right - left < 0.15 * width_px:
            continue
        points = []
        for column in (left, right):
            ray = inverse @ np.array([column, row, 1.0])
            denominator = normal @ ray
            if abs(denominator) < 1e-6:
                break
            points.append(ray / denominator)      # plane at unit distance
        if len(points) == 2:
            widths.append(np.linalg.norm(points[0] - points[1]))
    if not widths:
        return None
    return float(lane_width / np.median(widths))


def estimate_trajectory(frame_paths, intrinsics, masks=None, height=DEFAULT_CAMERA_HEIGHT,
                        lane_masks=None, drivable_masks=None, verbose=True):
    """Camera-to-world poses, (N, 4, 4), starting at the identity.

    :param frame_paths: frame images in temporal order
    :param intrinsics: 3x3 K the frames were taken with
    :param masks: optional per-frame boolean masks, True where the pixel is
        static scene; combined with the road band. Strongly recommended -- a
        vehicle being followed moves with the ego, so tracking it measures the
        gap closing rather than the ego's own motion and underestimates speed
        several-fold.
    :param drivable_masks: optional per-frame drivable-area masks
    :param height: camera height above the road, metres
    :param lane_masks: optional per-frame lane masks. When given, the height is
        measured from lane width and `height` is used only as the fallback.
    """
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    poses = [np.eye(4)]
    previous_gray = first
    failures = 0
    measured_heights = []

    for index in range(1, len(frame_paths)):
        gray = cv2.imread(str(frame_paths[index]), cv2.IMREAD_GRAYSCALE)
        static = masks[index - 1] if masks is not None else None
        drivable = drivable_masks[index - 1] if drivable_masks is not None else None
        # Pose is fitted over the whole static scene; only the scale step is
        # restricted to the road, so a clip whose road is briefly hidden still
        # gets its rotation and heading right.
        scene = np.ones(previous_gray.shape, bool) if static is None else static.astype(bool)
        plane = road_mask(previous_gray.shape, drivable=drivable, static=static)

        transform, inliers, normal = relative_pose(
            previous_gray, gray, scene, plane, intrinsics, height)
        if transform is None:
            failures += 1
            poses.append(poses[-1].copy())
        else:
            if lane_masks is not None and normal is not None:
                measured = camera_height_from_lanes(lane_masks[index - 1], intrinsics, normal)
                if measured is not None and 0.8 < measured < 2.5:
                    measured_heights.append(measured)
            poses.append(poses[-1] @ np.linalg.inv(transform))
        previous_gray = gray

    if measured_heights:
        median = float(np.median(measured_heights))
        if verbose:
            print(f"Lane width implies a camera height of {median:.2f} m "
                  f"(assumed {height:.2f} m, {len(measured_heights)} frames); "
                  f"scaling the trajectory by {median / height:.2f}x.")
        # Height enters linearly, so the whole trajectory rescales without
        # re-solving a single pair.
        for pose in poses:
            pose[:3, 3] *= median / height
    if verbose and failures:
        print(f"{failures} of {len(frame_paths) - 1} pairs could not be solved "
              "(held the previous pose).")
    return np.stack(poses)
