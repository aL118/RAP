"""Camera pitch, yaw and roll for a clip, measured without touching the point map.

    python calibrate_ground.py --output_dir ../data/test/wrongway/4
    python calibrate_ground.py --all --dataset test          # triage the whole set

Reports only; writes nothing. lift_frames_to_3d.py calls estimate() directly.

WHY NOT calibrate_clip.py

calibrate_clip.py fits the road plane in the UniDepth point map and reads pitch
off its slope. That measurement is downstream of the very thing it is trying to
correct: the point map's scale breathes 1.45x between adjacent frames on
wrongway/4 (check_depth_rigidity.py, which never fits a plane, confirms it
independently), so the per-frame fits disagree by more than a bolted camera can
and the verdict is always some flavour of "no single camera describes this clip".
That verdict is about the depth, not about the camera. A camera bolted to a
windscreen does have one pitch, and this module measures it from the image.

WHAT IT MEASURES

Painted lane markings are parallel lines on the road plane, so their images meet
at the vanishing point of the road direction. detect_lanes.py has already
segmented them, so the input is free. One VP per frame gives two of the three
angles in closed form:

  pitch   how far the VP sits below the principal point. This is the one that
          matters: on wrongway/4 the VP is at row 719 against cy = 529, i.e. the
          camera looks 7.0 deg down, and lift_frames_to_3d assumed 0.
  yaw     how far it sits to the side -- the camera's heading against the road's,
          which skews ego x/y and so the whole BEV.

Roll needs a second constraint, because one VP fixes a direction and a direction
is invariant to rotation about itself. The clip supplies it: as the road bends,
the per-frame VPs slide along the horizon line, and the line through them is the
horizon, whose tilt is the roll.

So the two side angles want opposite clips, and neither is free:

  yaw    needs a road that does NOT bend. A bend moves the VP sideways by far
         more than any mount is tilted -- 1525 px of column spread on wrongway/4
         -- and the median of that is the road's average heading, not the
         camera's. Held at 0 unless the clip is straight enough to mean it.
  roll   needs a road that DOES bend, and then some. It is believed only when
         the tilt explains substantially more vertical VP swing than the fit's
         own scatter; on wrongway/4 it explains 58 px against a 57 px residual,
         which is not a measurement, so that clip is held at 0 too.

Pitch survives both because the VP's row is the stable coordinate: a road's
grade changes far less than its heading does.

REFINING IT AGAINST THE OBJECTS

The VP is the vanishing point of the *road*, and that is the horizon only where
the road is flat and level with the camera. Ahead of a real clip it dips and
crests, and the median VP inherits whatever the clip did on average: on
wrongway/4 the lane VP lands at row 719 while the vehicles standing on the road
imply 691, and 28 px of horizon is a fifth of every range at 20 m.

The vehicles are the better witness, and they cost nothing extra. For a mask
standing on the road, the object's metric height is

    h_px * mount_height / (contact_row - horizon_row)

-- the focal length cancels -- so the horizon is the one row that makes a fleet
of cars come out car-height. That is a single robust scalar fitted over every
detection in the clip, it needs no point map either, and it is fitted against
the quantity the range is actually used for. estimate() reports both and uses
the refined one, so a disagreement between the two is visible rather than
silent.

WHAT IT CANNOT DO

Height. There is no metric reference anywhere in a monocular image, so the mount
height stays the assumption lift_frames_to_3d.CAMERA_HEIGHT already makes, and
the reconstruction is in units of that assumption. This is not a gap the ground
constraint leaves open: range consistency is what was broken, and consistency is
what the ground plane fixes. Absolute scale was never available and is not made
worse by pinning it here.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[0]

# Dashes are bridged before components are labelled, so a dashed centre line is
# one long component with a well-determined direction rather than a row of
# near-square blobs, each of whose direction is noise. Wider than tall for the
# same reason lanes.py's is: the gaps run along the marking.
DASH_BRIDGE_KERNEL = (25, 7)

# A component must clear both to contribute a line. The pixel floor drops
# speckle; the elongation (ratio of the covariance's principal axes) drops
# blobs, whose fitted direction is arbitrary and would drag the VP anywhere.
MIN_COMPONENT_PX = 60
MIN_ELONGATION = 3.0

# Markings within this of horizontal in the image are not the road direction --
# stop bars, crosswalk rungs and the leading edge of an arrow all run across it.
# They are near-perpendicular to the pencil of lines that meets at the VP, so
# including them does not just add noise, it pulls the solution off entirely.
MIN_LINE_TILT_DEG = 15.0

# Two lines determine a VP, but two nearly parallel ones determine it only in
# theory: the intersection slides along their shared direction under a pixel of
# noise. Require the frame's lines to span this much angle before believing it.
MIN_DIRECTION_SPREAD_DEG = 8.0
MIN_LINES = 2

# One reweighting pass, dropping lines the first solution missed by more than
# this. A marking on a side road, or a mask that bled onto a kerb, is a real
# line that belongs to a different pencil, and least squares alone would split
# the difference between the two.
VP_INLIER_PX = 60.0

# The VP must land in a plausible place before it is allowed to set the camera's
# attitude. Vertically this is the strong test -- a VP outside the middle band
# of the frame means the lines that made it were not the road. Horizontally it
# is loose, because a bend genuinely pushes the VP well outside the image.
VP_ROW_BAND = (0.25, 0.85)
VP_COL_MARGIN = 2.0            # multiples of the image width, either side

# Frames needed before a clip's median VP is worth reporting at all.
MIN_FRAMES = 8

# --- refining the horizon against the objects standing on the road ----------

# Detections needed before the refinement is attempted at all. It is one scalar
# fitted to a median, so it does not need many -- but a handful of vehicles all
# at one range would fit the horizon to their own shared error.
MIN_CONTACT_DETECTIONS = 40

# A mask shorter than this has too coarse a height for its implied metric height
# to mean anything: at 20 px, one pixel of segmentation slop is 5%.
MIN_CONTACT_HEIGHT_PX = 30

# How far the refinement may move the VP's pitch before it is refused. They are
# independent measurements of the same angle, so a small disagreement is the
# road's grade and a large one means one of them is not measuring the horizon.
MAX_REFINEMENT_DEG = 6.0

# Rows a contact must sit below the horizon to vote. Near the horizon the
# implied height goes as 1/(rows below it), so the last few rows carry enormous
# heights and would set the median on their own -- and they are the samples the
# ground range is not used for anyway (see lift_frames_to_3d.GROUND_MAX_RANGE).
# The cut is made once, against the VP's own horizon, so the objective is a
# median over a fixed set and stays monotonic in the pitch being searched.
#
# 60 px is ~39 m at this mount height and focal length, and the answer barely
# depends on it: swept over 20-180 px on wrongway/4 -- 849 contacts down to 134,
# a horizon of 116 m down to 13 m -- the refined pitch stayed within
# +5.60 to +6.11 deg. An independent estimate from regressing mask-bottom row
# against the point map's depth put it at +5.46 deg.
MIN_ROWS_BELOW_HORIZON = 60.0

# Range of pitches scanned, in degrees, and the step. Wider than any windscreen
# mount, because the scan only has to contain the answer -- it tolerates ends
# where the objective is undefined, which a bisection bracket does not.
REFINE_BRACKET_DEG = (-15.0, 25.0)
REFINE_SCAN_STEP_DEG = 0.1

# Roll comes from the line through the per-frame VPs, which is only a line if
# they moved. Below this spread (in pixels, p90 - p10 of the VP column) the
# clip is straight, the fit is unconditioned, and roll is reported as 0.
MIN_VP_SPREAD_PX = 150.0

# Spread is necessary and nowhere near sufficient. A cloud that is 1500 px wide
# and 200 px tall fits a line with a flattering eigenvalue ratio no matter what
# the roll is, because the width alone makes it look like a line. The test that
# means something compares the vertical swing the fitted tilt predicts across
# that width against the scatter about the fit: roll is a measurement only when
# it explains the cloud's shape by this factor more than the noise does.
MIN_ROLL_SNR = 3.0

# ... and even then, a horizon tilted past this is far more likely to be a bad
# pencil of lines than a bolted camera, so it is refused too.
MAX_ROLL_DEG = 10.0

# Above this VP column spread the road bends, and the median VP column is
# telling you where the road went rather than where the camera points. Yaw is
# held at 0 rather than absorbing the clip's average heading into the ego frame.
MAX_VP_SPREAD_FOR_YAW_PX = 400.0

# Attitudes past these are not a windscreen mount, they are a broken estimate.
MAX_PITCH_DEG = 25.0
MAX_YAW_DEG = 25.0

# cam -> ego for a perfectly level, forward-looking camera: camera axes are
# (x right, y down, z forward) and ego axes are (x forward, y left, z up), so
# this is the axis swap lift_frames_to_3d._camera_to_ego open-codes.
LEVEL_CAM_TO_EGO = np.array([[0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0]])


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class GroundCalibration:
    """The camera's attitude in the ego frame, and the transforms it implies.

    Angles are the camera's, in radians, and each is the physical tilt rather
    than the rotation that undoes it: `pitch` is positive looking down, `yaw`
    positive looking right of the vehicle's forward axis, `roll` positive with
    the horizon falling to the right. The default is the level camera
    lift_frames_to_3d assumed before this existed, so an un-calibrated clip
    keeps exactly its old geometry.
    """
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    height: float = 1.5
    frames: int = 0
    vanishing_point: tuple = None
    vp_spread_px: float = 0.0
    roll_snr: float = 0.0
    refinement_deg: float = None
    source: str = "assumed level"

    def cam_to_ego(self) -> np.ndarray:
        """3x3 rotation taking camera-frame directions to ego-frame ones.

        Each factor undoes one tilt, so the signs are the negatives of the
        angles above. Applied pitch-then-yaw outermost because that is the order
        they are solved in -- with a windscreen mount's angles the composition
        order moves the result by far less than the estimate's own spread.
        """
        return _rot_z(-self.yaw) @ _rot_y(-self.pitch) @ _rot_x(-self.roll) \
            @ LEVEL_CAM_TO_EGO

    def ego_to_camera(self) -> np.ndarray:
        """The 4x4 inverse, which is what boxes_3d.json carries for every reader."""
        rotation = self.cam_to_ego()
        transform = np.eye(4)
        transform[:3, :3] = rotation.T
        transform[:3, 3] = -rotation.T @ np.array([0.0, 0.0, self.height])
        return transform

    def camera_to_ego(self, points_cam: np.ndarray) -> np.ndarray:
        """(3, N) camera-frame points to (3, N) ego-frame ones."""
        return self.cam_to_ego() @ points_cam + \
            np.array([[0.0], [0.0], [self.height]])

    def describe(self) -> str:
        return (f"pitch {np.degrees(self.pitch):+.2f} deg  "
                f"yaw {np.degrees(self.yaw):+.2f} deg  "
                f"roll {np.degrees(self.roll):+.2f} deg  "
                f"height {self.height:.2f} m  ({self.source})")


def _component_lines(mask: np.ndarray) -> list:
    """One (point, unit direction) per lane marking long and thin enough to be one."""
    bridged = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, DASH_BRIDGE_KERNEL))
    count, labels = cv2.connectedComponents(bridged, connectivity=8)
    lines = []
    for label in range(1, count):
        rows, cols = np.nonzero(labels == label)
        if len(rows) < MIN_COMPONENT_PX:
            continue
        points = np.stack([cols, rows]).astype(np.float64)
        centre = points.mean(axis=1, keepdims=True)
        # Principal axis of the component: its direction, plus the elongation
        # that says whether "direction" means anything for this blob.
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points - centre))
        if eigenvalues[0] <= 1e-9 or eigenvalues[1] / eigenvalues[0] < MIN_ELONGATION ** 2:
            continue
        direction = eigenvectors[:, 1]
        if abs(np.degrees(np.arctan2(direction[1], direction[0]))) < MIN_LINE_TILT_DEG:
            continue
        lines.append((centre.reshape(2), direction / np.linalg.norm(direction),
                      float(len(rows))))
    return lines


def _solve_vanishing_point(lines) -> np.ndarray:
    """Least-squares meeting point of a pencil of lines, or None if unconditioned.

    Each line contributes its perpendicular offset, so the point minimises the
    summed squared distance to all of them -- the standard formulation, and the
    one that degrades gracefully when the lines nearly but do not exactly meet.
    """
    if len(lines) < MIN_LINES:
        return None
    angles = np.array([np.arctan2(d[1], d[0]) for _, d, _ in lines])
    # Folded to a half turn: a line has no head or tail, so directions pi apart
    # are the same direction and must not read as maximal spread.
    folded = np.sort(np.mod(angles, np.pi))
    if np.degrees(folded[-1] - folded[0]) < MIN_DIRECTION_SPREAD_DEG:
        return None
    normal_matrix = np.zeros((2, 2))
    rhs = np.zeros(2)
    for point, direction, weight in lines:
        normal = np.array([-direction[1], direction[0]])
        normal_matrix += weight * np.outer(normal, normal)
        rhs += weight * normal * float(normal @ point)
    if np.linalg.cond(normal_matrix) > 1e8:
        return None
    return np.linalg.solve(normal_matrix, rhs)


def _vanishing_point(mask: np.ndarray, image_hw) -> np.ndarray:
    """One frame's road vanishing point, or None if its markings cannot say."""
    lines = _component_lines(mask)
    point = _solve_vanishing_point(lines)
    if point is None:
        return None
    # Reweight once against the first solution: a marking belonging to a side
    # road is a real line in a different pencil, and averaging the two pencils
    # lands between them rather than on either.
    kept = [(p, d, w) for p, d, w in lines
            if abs(float(np.array([-d[1], d[0]]) @ (point - p))) <= VP_INLIER_PX]
    if len(kept) >= MIN_LINES:
        refined = _solve_vanishing_point(kept)
        point = refined if refined is not None else point

    height, width = image_hw
    if not VP_ROW_BAND[0] * height <= point[1] <= VP_ROW_BAND[1] * height:
        return None
    if abs(point[0] - width / 2.0) > VP_COL_MARGIN * width:
        return None
    return point


def frame_vanishing_points(lane_masks_dir: Path, image_hw) -> np.ndarray:
    """Every frame's VP, as an (N, 2) array of pixel coordinates."""
    points = []
    for mask_path in sorted(lane_masks_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = mask > 127
        if mask.shape != tuple(image_hw):
            mask = cv2.resize(mask.astype(np.uint8), (image_hw[1], image_hw[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue
        point = _vanishing_point(mask, image_hw)
        if point is not None:
            points.append(point)
    return np.array(points) if points else np.zeros((0, 2))


def _roll_from_horizon(points: np.ndarray):
    """(camera roll, signal-to-noise) from the line the per-frame VPs trace out.

    Total least squares rather than a regression of row on column: both
    coordinates are noisy, and on a clip whose VPs barely move the ordinary fit
    would report the noise's own slope with confidence.

    Returns (None, snr) when the fit does not clear its gates, so a caller can
    still report how close it came.
    """
    if len(points) < MIN_FRAMES:
        return None, 0.0
    spread = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10))
    if spread < MIN_VP_SPREAD_PX:
        return None, 0.0
    centred = points - points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    direction = eigenvectors[:, 1]
    roll = float(np.arctan2(direction[1], direction[0]))
    # Folded to the shallow representative: the horizon has no left or right end.
    roll = (roll + np.pi / 2) % np.pi - np.pi / 2

    # What the tilt claims to explain, against what it leaves unexplained.
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.median(np.abs(centred @ normal)))
    swing = abs(np.tan(roll)) * spread
    snr = swing / max(residual, 1e-6)
    if snr < MIN_ROLL_SNR or abs(np.degrees(roll)) > MAX_ROLL_DEG:
        return None, snr
    return roll, snr


def _implied_height_ratio(pitch: float, contacts: np.ndarray,
                          intrinsics: np.ndarray, height: float) -> float:
    """Median log(implied object height / its class prior) at this pitch.

    Zero when the fleet comes out the size the priors say it is. Rises with
    pitch -- a lower horizon means fewer rows between it and each contact point,
    so a longer range and a taller object -- which is what lets the caller find
    the answer by looking for the sign change.

    `contacts` is an (N, 3) array of (top row, bottom row, prior height). None
    when too few of them are below the horizon this pitch implies to speak.
    """
    horizon = intrinsics[1, 2] + intrinsics[1, 1] * np.tan(pitch)
    rows_below = contacts[:, 1] - horizon
    # Above the horizon at this pitch: no intersection with the road, so the
    # sample cannot vote. Dropped rather than clamped, so the objective stays a
    # median over real samples at every pitch it is evaluated at.
    usable = rows_below > 1.0
    if usable.sum() < MIN_CONTACT_DETECTIONS:
        return None
    implied = ((contacts[usable, 1] - contacts[usable, 0] + 1.0) * height
               / rows_below[usable])
    return float(np.median(np.log(implied / contacts[usable, 2])))


def refine_pitch(pitch: float, contacts, intrinsics: np.ndarray, height: float):
    """(refined pitch, its shift in degrees), or (pitch, None) if it cannot run.

    A scan for the sign change rather than a bisection between two fixed ends:
    the objective is undefined wherever the candidate horizon drops below the
    image, which a wide bracket's upper end always does, and a bracket narrow
    enough to avoid that is not guaranteed to contain the answer.

    Everything about it is a median, so a mask that leaked onto a shadow or a
    vehicle parked on a kerb moves the result by nothing at all.
    """
    horizon = intrinsics[1, 2] + intrinsics[1, 1] * np.tan(pitch)
    near = np.array([c for c in contacts if c[1] - horizon >= MIN_ROWS_BELOW_HORIZON],
                    dtype=np.float64)
    if len(near) < MIN_CONTACT_DETECTIONS:
        return pitch, None

    grid = np.radians(np.arange(REFINE_BRACKET_DEG[0], REFINE_BRACKET_DEG[1],
                                REFINE_SCAN_STEP_DEG))
    values = [(angle, _implied_height_ratio(angle, near, intrinsics, height))
              for angle in grid]
    values = [(angle, value) for angle, value in values if value is not None]
    crossing = next(((values[i][0], values[i + 1][0])
                     for i in range(len(values) - 1)
                     if values[i][1] < 0.0 <= values[i + 1][1]), None)
    if crossing is None:
        return pitch, None

    low, high = crossing
    for _ in range(40):
        middle = 0.5 * (low + high)
        value = _implied_height_ratio(middle, near, intrinsics, height)
        if value is None:
            break
        if value < 0.0:
            low = middle
        else:
            high = middle
    refined = 0.5 * (low + high)
    shift = float(np.degrees(refined - pitch))
    if abs(shift) > MAX_REFINEMENT_DEG:
        return pitch, shift
    return refined, shift


def _attitude_from_vp(point, intrinsics: np.ndarray, roll: float):
    """(pitch, yaw) putting the VP's ray straight down the ego x axis.

    The VP is the image of the road's direction, and the ego frame is defined so
    the road runs along +x. So the two angles are whatever it takes to rotate
    that one ray onto that one axis, which is a closed form rather than a fit:
    pitch kills its vertical component, yaw the lateral one that is left.
    """
    ray = np.linalg.solve(intrinsics, np.array([point[0], point[1], 1.0]))
    level = LEVEL_CAM_TO_EGO @ ray
    rolled = _rot_x(-roll) @ level
    pitch = -float(np.arctan2(rolled[2], rolled[0]))
    forward = float(np.hypot(rolled[0], rolled[2]))
    yaw = -float(np.arctan2(-rolled[1], forward))
    return pitch, yaw


def contacts_from_masks(masks_json, image_hw, priors) -> list:
    """(top row, bottom row, prior height) per detection worth refining against.

    Decoding is the caller's job -- this file has no business importing
    pycocotools for one field -- so `masks_json` is taken already decoded, as
    (top, bottom, category) triples.
    """
    height = image_hw[0]
    contacts = []
    for top, bottom, category in masks_json:
        prior = priors.get(category)
        if prior is None or bottom >= height - 3:
            continue
        if bottom - top + 1 < MIN_CONTACT_HEIGHT_PX:
            continue
        contacts.append((float(top), float(bottom), float(prior)))
    return contacts


def estimate(lane_masks_dir: Path, intrinsics: np.ndarray, image_hw,
             height: float = 1.5, contacts=None) -> GroundCalibration:
    """The clip's calibration, or the level default when it cannot be measured.

    Falling back rather than raising, and saying so in `source`, because a clip
    with no lane markings is a normal thing to lift -- it just lifts with the
    geometry it would have had before this module existed.

    `contacts` is contacts_from_masks()' output; without it the pitch is the
    lane VP's alone.
    """
    if lane_masks_dir is None or not Path(lane_masks_dir).is_dir():
        return GroundCalibration(height=height, source="no lane masks")
    points = frame_vanishing_points(Path(lane_masks_dir), image_hw)
    if len(points) < MIN_FRAMES:
        return GroundCalibration(height=height, frames=len(points),
                                 source=f"only {len(points)} usable frames")

    # Median rather than mean over the clip: a frame whose pencil was polluted
    # by a side road gives a VP that is wrong by hundreds of pixels, not by tens.
    median = np.median(points, axis=0)
    spread = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10))
    roll, roll_snr = _roll_from_horizon(points)
    pitch, yaw = _attitude_from_vp(median, intrinsics, roll or 0.0)

    # Yaw is thrown away on a bending clip rather than trusted -- see
    # MAX_VP_SPREAD_FOR_YAW_PX. Pitch is kept either way.
    held = []
    if spread > MAX_VP_SPREAD_FOR_YAW_PX:
        yaw = 0.0
        held.append("yaw held at 0: road bends")
    if roll is None:
        held.append("roll held at 0: unconditioned" if roll_snr == 0.0
                    else f"roll held at 0: snr {roll_snr:.1f}")

    refinement = None
    if contacts:
        refined, refinement = refine_pitch(pitch, contacts, intrinsics, height)
        # refine_pitch reports the shift even when it refuses to apply it, so
        # `refined` is only adopted below when the shift cleared the limit.
        if refinement is not None and abs(refinement) > MAX_REFINEMENT_DEG:
            held.append(f"pitch refinement {refinement:+.1f} deg refused")
        else:
            pitch = refined

    if abs(np.degrees(pitch)) > MAX_PITCH_DEG or abs(np.degrees(yaw)) > MAX_YAW_DEG:
        return GroundCalibration(height=height, frames=len(points),
                                 vanishing_point=tuple(median), vp_spread_px=spread,
                                 roll_snr=roll_snr, refinement_deg=refinement,
                                 source="rejected: attitude out of range")
    source = "lane VP"
    if refinement is not None and abs(refinement) <= MAX_REFINEMENT_DEG:
        source += f" refined {refinement:+.2f} deg on {len(contacts)} contacts"
    if held:
        source += " (" + "; ".join(held) + ")"
    return GroundCalibration(
        pitch=pitch, yaw=yaw, roll=roll or 0.0, height=height, frames=len(points),
        vanishing_point=(float(median[0]), float(median[1])), vp_spread_px=spread,
        roll_snr=roll_snr, refinement_deg=refinement, source=source)


def _clip_intrinsics(output_dir: Path):
    """The camera boxes_3d.json already carries, so the CLI reports what the lift
    would actually use rather than re-deriving it from the point maps."""
    boxes_path = Path(output_dir) / "boxes_3d.json"
    if not boxes_path.is_file():
        return None, None
    import json
    data = json.loads(boxes_path.read_text())
    for entry in data.values():
        camera = entry.get("camera")
        if camera:
            return (np.asarray(camera["intrinsics"], dtype=np.float64),
                    tuple(camera["image_hw"]))
    return None, None


def report(name: str, output_dir: Path, height: float, verbose: bool) -> None:
    intrinsics, image_hw = _clip_intrinsics(output_dir)
    if intrinsics is None:
        print(f"{name:<22} -- no boxes_3d.json to read the camera from")
        return
    lane_masks_dir = Path(output_dir) / "lane_masks"
    # Imported here, not at module scope: lift_frames_to_3d imports this file,
    # and it is the one that already has pycocotools open.
    contacts = None
    masks_path = Path(output_dir) / "mask_results_preds.json"
    if masks_path.is_file():
        import json
        import lift_frames_to_3d
        contacts = lift_frames_to_3d._contact_rows(
            json.loads(masks_path.read_text()), image_hw)
    calibration = estimate(lane_masks_dir, intrinsics, image_hw, height, contacts)
    print(f"{name:<22} {calibration.frames:>3} frames  {calibration.describe()}")
    if not verbose:
        return
    if calibration.vanishing_point is None:
        print("    -> no usable vanishing point; the lift would run level, as before.")
        return
    print(f"    vanishing point  ({calibration.vanishing_point[0]:7.1f}, "
          f"{calibration.vanishing_point[1]:7.1f}) px   "
          f"principal point ({intrinsics[0, 2]:7.1f}, {intrinsics[1, 2]:7.1f}) px")
    print(f"    VP column spread {calibration.vp_spread_px:6.1f} px   "
          f"(yaw needs < {MAX_VP_SPREAD_FOR_YAW_PX:.0f}, roll needs > {MIN_VP_SPREAD_PX:.0f})")
    print(f"    roll signal/noise {calibration.roll_snr:5.1f}          "
          f"(needs > {MIN_ROLL_SNR:.1f})")
    if calibration.refinement_deg is not None:
        verdict = "refused" if abs(calibration.refinement_deg) > MAX_REFINEMENT_DEG \
            else "applied"
        print(f"    object contacts move the pitch "
              f"{calibration.refinement_deg:+.2f} deg -- {verdict} "
              f"(limit {MAX_REFINEMENT_DEG:.0f} deg)")
    horizon = intrinsics[1, 2] + intrinsics[1, 1] * np.tan(calibration.pitch)
    print(f"    calibrated horizon row {horizon:7.1f} px, "
          f"{horizon - intrinsics[1, 2]:+.0f} px from where a level camera puts it "
          f"-- that offset is the {np.degrees(calibration.pitch):+.2f} deg of pitch.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", help="run dir holding lane_masks/ and boxes_3d.json")
    ap.add_argument("--all", action="store_true", help="triage every clip in --dataset")
    ap.add_argument("--dataset", default="test")
    ap.add_argument("--run", default="4")
    ap.add_argument("--camera_height", type=float, default=1.5,
                    help="assumed mount height above the road, metres. Not measured "
                         "here and not measurable from one camera; it is the unit the "
                         "reconstruction is expressed in.")
    args = ap.parse_args()

    if args.all:
        data = ROOT.parent / "data" / args.dataset
        for clip in sorted(p for p in data.iterdir() if p.is_dir()):
            run_dir = clip / args.run if args.run else clip
            if (run_dir / "lane_masks").is_dir():
                report(clip.name, run_dir, args.camera_height, False)
        return
    if not args.output_dir:
        ap.error("give --output_dir, or --all to triage a dataset")
    path = Path(args.output_dir).resolve()
    if not (path / "lane_masks").is_dir():
        raise SystemExit(f"error: no lane masks at {path / 'lane_masks'}; "
                         "run detect_lanes.py first")
    name = f"{path.parent.name}/{path.name}" if path.name.isdigit() else path.name
    report(name, path, args.camera_height, True)


if __name__ == "__main__":
    main()
