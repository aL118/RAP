"""
Lifts infer_frames.py's 2D mask detections into 3D oriented boxes usable by
visualization/raster_frames.py, by backprojecting each mask's pixels through
the cached UniDepth point map (see infer_unidepth_on_frames.py) and fitting a
ground-plane-constrained oriented box to the resulting point cloud.

Unlike the nuScenes pipeline there is no ego pose, camera calibration, or
multi-camera aggregation for nightcrash footage, so this skips straight from
UniDepth's per-pixel camera-frame point map to a mask-restricted point cloud
-- the map is already pixel-aligned to the mask, so unlike
load_pseudolidar_nuscenes.py's reprojection (needed only because its points
get round-tripped through a multi-camera "global frame" aggregation first)
there's no intrinsics-based reprojection step here. Points are axis-remapped
from camera convention (x-right, y-down, z-forward) to ego convention
(x-forward, y-left, z-up) before fitting, since the box format
[x, y, z, l, w, h, heading] assumes z is vertical.

Each frame's entry also carries the camera the boxes were fit through (see
_camera_intrinsics), so raster_frames.py can render them back through the
*same* camera. Rendering through a different one (e.g. navsim's fixed CAM_F0
rig) puts the boxes nowhere near the masks they came from -- the error is
worst for near objects, where a sub-metre camera translation is a large
fraction of the object's depth.

Traffic lights are lifted separately, into each frame's "traffic_lights" list
rather than its "boxes": they are drawn as fixed-size upright cuboids
colour-coded by their lit state, so what has to be recovered for them is a
position and a state, not an oriented extent (see traffic_lights.py).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

import box_schema
from lanes import lane_polylines, stabilise_lanes
import calibrate_ground
import lanes as lanes_module
from traffic_lights import TRAFFIC_LIGHT_DIMS, classify_state, is_traffic_light

ROOT = Path(__file__).resolve().parents[0]

# Approximate dashcam mount height above the ground, in metres. There's no
# real calibration for nightcrash footage to do better than a constant here.
# Note this is purely a choice of where to put the ego frame's origin: the
# rendered overlay is invariant to it, because raster_frames.py places its
# camera at the matching height (see _ego_to_camera_transform).
CAMERA_HEIGHT = 1.5

# Need more than this many pixels-with-depth in a mask before attempting to
# fit a box -- mirrors load_pseudolidar_nuscenes.py's `shape[1] <= 3` guard.
MIN_MASK_POINTS = 6

# Half-width of the depth band kept around a mask's median depth, as a
# fraction of that median. Monocular depth is least reliable near object
# boundaries, where it ramps smoothly onto the background -- there is no gap
# to cluster on, so gap/DBSCAN-style rejection does not catch it, and a MAD
# threshold is defeated by the outliers inflating the MAD itself (observed on
# this footage: a mask whose own MAD was 4.7 m, so a 3.5x-MAD cut kept points
# spanning 6-27 m and fitted a 21 m "car"). The band is relative, not
# absolute, because monocular depth error scales with depth.
DEPTH_BAND_FRAC = 0.25

# Mask erosion before sampling points, as a fraction of the mask's smaller
# bbox side. The outermost ring of a segmentation mask is where the depth map
# interpolates between object and background, so those pixels contribute
# points strung out along the viewing ray. Skipped if it would leave too few
# points (thin/distant masks).
MASK_EROSION_FRAC = 0.06

# Percentile used for the box's extents along each fitted axis (so the box
# spans PCTL..100-PCTL rather than min..max). Shaves the residual stragglers
# that survive the depth band without needing a second clustering pass.
EXTENT_PERCENTILE = 2.0

# Rough real-world (width, height) in metres per class, used *only* to sanity
# check the geometry at the end of a run -- never to constrain a fit.
#
# A mask's pixel bbox, its median depth and the intrinsics over-determine an
# object's physical size, so comparing that implied size against a prior is the
# one check that can catch a wrong camera model. Nothing else can: the focal
# length is never used in isolation, so an incorrect one stays self-consistent
# all the way through (_intrinsics_from_point_map even recovers it *from* the
# point map, so it agrees with the depth by construction, and raster_frames.py
# renders back through the same K). The only visible symptom is that every
# object comes out uniformly too small, which reads as a fitting problem rather
# than a calibration one. Observed on wide-angle dashcam footage: UniDepth
# predicted a 50 deg FOV for a ~113 deg lens, and every box came out ~3x too
# small in cross-section while its depth stayed plausible.
CLASS_SIZE_PRIOR = {           # metres, (width, height)
    "car": (1.8, 1.5),
    "truck": (2.5, 3.2),
    "bus": (2.6, 3.2),
    "van": (2.0, 2.2),
    "person": (0.6, 1.7),
    "pedestrian": (0.6, 1.7),
    "bicycle": (0.6, 1.1),
    "motorcycle": (0.8, 1.4),
}

# Implied/prior size ratios outside this band trip the run-end warning. Wide,
# because the priors are class averages and a mask's bbox includes whatever
# pose the object is in -- this is meant to catch a 3x scale error, not a 20% one.
SIZE_RATIO_BOUNDS = (0.7, 1.4)

# A heading counts as "collapsed" when it lands this close to the object's own
# bearing from the camera. Depth noise smears a mask's points along the viewing
# ray; once that smear exceeds the object's own size, _fit_box's minAreaRect
# elects the ray as the long axis and the heading degenerates to the line of
# sight -- the box becomes a rod pointed at the camera, which reprojects far
# too small and slides off its object as the fit wobbles. Headings drawn
# uniformly would land inside this band about 11% of the time.
HEADING_COLLAPSE_DEG = 10.0

# 1 = resolve one camera for the whole clip rather than one per frame.
#
# UniDepth predicts intrinsics per image, and on a clip that is one continuous
# shot from one bolted-down camera it should predict the same ones every time.
# It does not: measured across CARE_YTB, fx wanders 1296-2458 within
# flour_explode, 1512-2327 within buick_nearmiss, 1496-2297 within honda_swerve.
# Every frame is then lifted through a different camera, so a stationary object
# changes size and position frame to frame for no reason in the scene -- and
# smooth_boxes cannot repair that, because it is re-fitting geometry that moved
# underneath it rather than an object that moved.
#
# Taking the median over the clip does not make the camera correct (that needs a
# calibration this footage does not have, and CLASS_SIZE_PRIOR is what reports
# how wrong it is); it makes it *consistent*, which is the half of the problem
# that can be fixed without one. 0 restores the per-frame behaviour.
CLIP_INTRINSICS = True

# How many frames to recover intrinsics from before taking the median. Every
# frame would be exact but reads a 25 MB point map each; the median of 20 evenly
# spaced frames is stable to well under the spread being corrected, and this
# pass is pure extra I/O on top of a stage that already reads them all once.
CLIP_INTRINSICS_SAMPLE = 20

# Below this many detections a class's ratios are too noisy to report on.
MIN_SANITY_SAMPLES = 20

# Two detections in the same frame whose masks overlap by more than this are
# taken to be the same object, and only the higher-scoring one is lifted.
#
# The detector emits the same mask twice under different labels -- frame 80 of
# the beepbeep clip carries a truck (0.39) and a car (0.33) at mask IoU 1.00.
# Both get lifted, and because the box's proportions come from its class prior
# they land as two very differently shaped boxes on one vehicle, which reads as
# the box flipping from lying down to standing up between frames. Suppression is
# class-agnostic for exactly that reason: a per-class NMS inside the detector
# cannot see the pair, since they belong to different classes.
DUPLICATE_MASK_IOU = 0.7

# ... and a mask this far *inside* a higher-scoring one is taken to be the same
# object too, even when their IoU is low. A nested pair has a small intersection
# relative to the union, so IoU alone never sees it: a leaky mask that swallowed
# a neighbour scored IoU 0.42 against the tight mask it contained, well under
# DUPLICATE_MASK_IOU, and both were lifted -- drawing a box inside a box. 24 such
# pairs in the beepbeep clip, against 26 that IoU already catches.
DUPLICATE_MASK_CONTAINMENT = 0.8


# A detection whose mask spans this fraction of the image width *and* reaches
# the bottom edge *and* recurs unchanged elsewhere in the clip is the ego's own
# bonnet, and is never lifted.
#
# The detector has no idea the camera is bolted to a car, so it labels the hood
# in front of the lens a "car" like any other. On changelane that fired in 135
# frames, and because the mask is a band across the whole frame with no depth
# structure to speak of, the fit came back as a 20 x 8 x 6.7 m "car" centred
# 6.3 m under the road -- a slab painted across the bottom of those frames, and
# (correctly, given its geometry) sorted in front of every real vehicle.
#
# The first two tests are geometric rather than a size sanity check, because the
# size is only a symptom: what identifies the bonnet is that it subtends the
# entire horizontal field of view while resting on the bottom edge. Nothing the
# camera is merely *looking* at can do that -- a 1.8 m car would have to be
# 2.5 m away, and would then tower up the frame rather than sit in a band along
# the bottom.
#
# Those two alone are not enough, and the case that breaks them is the one that
# matters most in this footage. When a vehicle hits the ego, SAM merges it with
# the bonnet into a single full-width, bottom-touching mask -- so a purely
# geometric test deletes the struck car at the exact frames of the collision,
# which is the event the clip exists to show. On changelane that is frames
# 83-91, where the black car coming across the lane is only ever detected merged
# into the bonnet.
#
# EGO_MASK_MIN_RECURRENCE is what separates them, and the discriminator is
# motion: the bonnet is rigidly attached to the camera, so its mask is the same
# pixels in every frame it appears in, while a struck vehicle moves and deforms
# and is gone within a second. Measured on changelane, at
# EGO_MASK_RECURRENCE_IOU: every one of the 126 genuine bonnet masks matches 125
# others, and each of the 9 collision masks matches at most 1. A clip whose
# bonnet is detected fewer than EGO_MASK_MIN_RECURRENCE times keeps it, which is
# the right way round to fail -- a spurious slab is a blemish, a deleted crash
# is the whole point of the clip.
EGO_MASK_WIDTH_FRACTION = 0.9
EGO_MASK_BOTTOM_MARGIN_PX = 2
EGO_MASK_RECURRENCE_IOU = 0.95
EGO_MASK_MIN_RECURRENCE = 5


# A detection that is all three of these at once is a piece of the scene the
# detector has labelled a vehicle, and is never lifted:
#   - sprawling: bigger than SURFACE_MASK_AREA_FRACTION of the frame,
#   - flat: wider than SURFACE_MASK_ASPECT times its own height,
#   - hollow: filling less than SURFACE_MASK_FILL of its own bounding box.
#
# The case is the wet asphalt between the ego and the traffic ahead, which
# GroundingDINO returns as a "car" at score 0.30-0.34 for the last 17 frames of
# changelane -- a 750 x 220 px sheet of road that lifts into a 4.9 x 2.0 x 1.6 m
# box floating over the lane in front of the bonnet.
#
# All three conditions are needed because each one alone throws away real
# vehicles on this footage. Hollow alone takes the ragged partial masks of
# distant cars (fill 0.23-0.31 at 1-5 k px). Flat alone takes a car overtaking
# at arm's length, seen side-on across 1016 px (frame 92). Sprawling alone takes
# every close truck. Together they describe something no vehicle can be: a large
# region that is much wider than it is tall *and* mostly empty inside its own
# bounding box -- i.e. a surface wrapping around the things standing on it,
# which is exactly what the road does and what a vehicle's own silhouette never
# does.
#
# Depth cannot referee this, which is why the test is 2D. The obvious check --
# do these points lie in the road plane rather than above it? -- fails on
# UniDepth's output: the road mask's lifted points straddle z = 0 by +-1 m and
# real cars come back with median heights as low as -1.8 m, so the two
# populations overlap completely.
#
# A mask reaching within SURFACE_MASK_BOTTOM_CLEARANCE of the bottom edge is
# exempt whatever its shape, because that is where a collision appears: a
# vehicle striking the ego merges with the bonnet into a wide, bottom-touching
# mask whose fill (0.42 on changelane frame 86) sits close enough to
# SURFACE_MASK_FILL to be at risk. Deleting a blemish is worth a shape
# heuristic; deleting the moment of impact is not, so the heuristic is not
# allowed to reach there. The road the detector does latch onto is bounded below
# by the ego's own bonnet and clears the edge by 134 px, so nothing is lost --
# and on a clip with no bonnet in shot the exemption merely lets a road blob
# through, which is the harmless direction.
#
# Measured: fires on those 17 frames and on nothing else across changelane,
# beepbeep, redlight and snowcrash (9485 detections).
SURFACE_MASK_AREA_FRACTION = 0.02
SURFACE_MASK_ASPECT = 2.5
SURFACE_MASK_FILL = 0.40
SURFACE_MASK_BOTTOM_CLEARANCE = 0.05   # of image height


# Anchor every fitted box to its own 2D detection: the rendered cuboid is
# required to cover that detection's mask bounding box and is never allowed to
# shrink inside it. Set False to keep the raw point-cloud fit.
#
# The point-cloud fit alone has no such invariant, and breaks it in two
# independent ways -- a wrong focal length shrinks every back-projected
# cross-section uniformly, and _reject_depth_outliers' relative band slices a
# horizontal stripe out of any object whose own depth extent exceeds the band
# (close, large objects especially: depth rises with image row because an
# object's far end projects toward the vanishing point, so a cut in depth is a
# cut in height). Neither is recoverable from the point cloud, because both
# leave it self-consistent. The mask, by contrast, is a direct measurement of
# where the object is on screen, so anchoring to it fixes the placement and the
# apparent size regardless of what the depth did.
ANCHOR_BOXES_TO_MASK = True

# Depth percentile taken as the object's visible near surface, which is what
# its mask actually corresponds to. Not the median: for a close object the
# median sits partway through its depth extent, so a box anchored there juts
# out in front of the object it is meant to bound.
NEAR_SURFACE_PERCENTILE = 20.0

# --- Range from the road rather than from the depth map ---------------------
#
# GROUND_CONTACT_RANGE is the answer to the measurement above being the wrong
# one to trust. NEAR_SURFACE_PERCENTILE reads an object's range out of the
# UniDepth point map, and that map's scale breathes: measured on wrongway/4, the
# whole scene dilates by up to 1.64x and down to 0.68x between adjacent frames
# (all boxes in a frame by the same factor -- check_depth_rigidity.py confirms
# 1.45x from LK correspondences alone). smooth_boxes.py cannot filter it out,
# because a shared per-frame scale makes every track agree and so looks like
# signal to a per-track filter.
#
# A vehicle standing on the road has a range that does not need the depth map at
# all: the row where its mask meets the road, the camera's height above that
# road, and the pitch calibrate_ground measured determine it outright. Measured
# on wrongway/4 across 22 tracks, frame-to-frame jitter (sd of the log step):
#
#   monocular range from the point map     0.087
#   mask height, i.e. any size-based fit   0.073     <- the ceiling that route has
#   ground contact row                     0.026
#
# The contact row wins because a segmentation boundary is far steadier where the
# wheels meet the tarmac than along a roofline against sky and branches. Note
# what this does and does not fix: it makes range *consistent*, which is what was
# broken. Absolute scale still rides on CAMERA_HEIGHT, which no monocular clip
# can measure -- see calibrate_ground's closing note.
GROUND_CONTACT_RANGE = True

# Rows above the mask's lowest one to take the contact column from. The single
# lowest pixel is a corner of the segmentation, not the middle of the tyre
# contact; a shallow band's median column is the same point without the noise.
CONTACT_BAND_PX = 3

# Beyond this the constraint stops being one. Range goes as 1/(rows below the
# horizon), so at the far end a pixel of segmentation noise -- or a tenth of a
# degree of pitch error -- is worth tens of metres, and the point map, for all
# its breathing, is the better guess. 60 m is where a 1 px error passes 10%.
GROUND_MAX_RANGE = 60.0

# The check that the contact row is a contact row. Given the ground range, the
# mask's pixel height implies a metric height for the object; against its class
# prior that ratio must be sane. It is not a size test -- it is how an occluded
# bottom edge (a car behind a car reads as standing where the nearer one does)
# and a mask that leaked onto a shadow announce themselves. Wide, because the
# prior is a class average and CLASS_SIZE_PRIOR's own band is 0.7-1.4 on data
# that was already right; this only has to catch the gross failures.
GROUND_SIZE_BAND = (0.45, 2.2)

# Fallback length in metres along the heading, per class. The fitted length is
# unusable whenever the heading has collapsed onto the line of sight, because
# it is then the depth smear rather than the object -- see HEADING_COLLAPSE_DEG.
CLASS_LENGTH_PRIOR = {
    "car": 4.5, "truck": 7.0, "bus": 11.0, "van": 5.2,
    "person": 0.6, "pedestrian": 0.6, "bicycle": 1.7, "motorcycle": 2.0,
}

# Verify-and-fit passes after the box is constructed. The construction makes the
# box's *front face* project onto the mask bbox, but the silhouette is the whole
# cuboid, so an oblique box over-covers by the flare of its side faces. These
# passes converge the silhouette onto the bbox instead, then a final pass
# guarantees containment.
COVER_ITERATIONS = 6

# Fraction of over-coverage left in place, so rounding in the projection cannot
# push the silhouette back inside the mask bbox.
COVER_MARGIN = 1.02

# Floor on the length pulled in to remove over-coverage, as a fraction of the
# class prior. Length is the knob for over-coverage because it is the uncertain
# quantity -- it runs along a heading that has usually collapsed onto the line
# of sight, whereas the cross-section is back-projected straight from the mask
# and is therefore the one measurement worth keeping. The floor stops a box
# from being flattened into a plate to satisfy the fit.
LENGTH_FLOOR_FRAC = 0.35

# Re-estimate each box's heading from the shape of its own mask, instead of
# keeping _fit_box's (which collapses onto the line of sight for most boxes --
# see HEADING_COLLAPSE_DEG -- so every car renders face-on however it is really
# parked). A cuboid seen off-axis projects to a hexagon whose shape is a strong
# function of yaw, so scoring that silhouette against the mask recovers the yaw.
#
# Crucially this works even though the metric scale is wrong: the search fits a
# free uniform scale per candidate, so only the silhouette's *shape* is compared,
# and shape is scale-invariant. Orientation is recoverable from a wrong-focal-
# length point map; absolute size is not.
HEADING_FROM_MASK = True

# Candidate yaws, spread over pi (a box is symmetric under a half turn, and the
# heading is only defined up to +-pi anyway).
HEADING_SEARCH_STEPS = 36

# Weight of the prior that a vehicle is parallel to the ego's own heading, i.e.
# yaw ~ 0 in the ego frame -- not yaw ~ bearing, because everything on a straight
# road is parallel to the road including us, whatever direction it lies in.
#
# Needed because the silhouette likelihood is weak on its own: with candidates
# properly aligned the IoU spread across a full half-turn of yaw is only about
# 0.10, comparable to the noise from mask roughness, so the raw argmax lands on
# arbitrary angles (a van driving straight away scored best at -45 deg). The
# prior costs a candidate up to this fraction of its IoU at a right angle and
# nothing at all when parallel, so a genuinely oblique vehicle still wins if the
# silhouette prefers it by more than a few percent -- but a flat curve resolves
# to "parallel to the road" instead of to noise.
#
# Headings are therefore prior-dominated and should not be read as measurements.
HEADING_PRIOR_WEIGHT = 0.15

# Side of the square grid the mask and the candidate silhouette are rasterised
# onto to score their overlap. 64 is well below any mask's own resolution but
# far finer than the 5 deg yaw quantisation, so it is not the limiting factor.
SILHOUETTE_GRID = 64

# The relaxed wrapping constraint: the silhouette must still cover at least this
# fraction of the mask's pixels, but is no longer required to contain the mask's
# bounding box. Containment is too strong once orientation is in play -- the
# bounding box of a hexagonal silhouette is much larger than the hexagon, so
# demanding it forces the box back to face-on and oversized (the same reason a
# ball's bounding box is a square while a cube's silhouette is a hexagon).
MIN_MASK_COVERAGE = 0.90

# Multiplier per shrink step, applied while MIN_MASK_COVERAGE still holds.
SHRINK_STEP = 0.94

# A mask bbox edge this close to the frame border is treated as truncation --
# the frame cut the object off there -- rather than as its real extent.
#
# Truncated masks otherwise wreck the fit in two ways at once. Their aspect
# ratio is not the object's (a truck clipped to a sliver at the left edge
# measured 218x747, aspect 0.29), so sizing the box to span both dimensions
# inflates the unclipped one several fold; and their centre is not the object's
# centre, so centring on it slides the box inboard across whatever is parked
# behind. Both are fixed by fitting and aligning only against edges the frame
# did not cut, and letting the box run off-screen on the side that was cut --
# which is where the object actually continues.
TRUNCATION_MARGIN_PX = 2


# The clip's camera attitude, set once by lift_frames() before anything is
# lifted and read by every transform below. A module-level value rather than a
# threaded argument because _camera_to_ego is called from the box fit, the
# anchoring, the traffic-light lift and the lane lift, and the alternative is
# five signatures carrying a constant. It defaults to the level camera this file
# assumed before calibrate_ground existed, so a clip with no lane masks -- or a
# caller that never sets it -- lifts with exactly the old geometry.
_CALIBRATION = calibrate_ground.GroundCalibration(height=CAMERA_HEIGHT)


def set_calibration(calibration) -> None:
    """Installs the clip's measured camera attitude. Call before lifting."""
    global _CALIBRATION
    _CALIBRATION = calibration


def calibration():
    """The attitude currently in force, for the passes that need it explicitly."""
    return _CALIBRATION


def _camera_to_ego(points_cam: np.ndarray) -> np.ndarray:
    """points_cam: (3, N) camera convention (x-right, y-down, z-forward).
    Returns (3, N) ego convention (x-forward, y-left, z-up)."""
    return _CALIBRATION.camera_to_ego(points_cam)


def _ego_to_camera_transform() -> np.ndarray:
    """The 4x4 inverse of _camera_to_ego, for raster_frames.py to render with."""
    return _CALIBRATION.ego_to_camera()


def _intrinsics_from_point_map(xyz: np.ndarray) -> np.ndarray:
    """Recovers the pinhole intrinsics UniDepth used, from the point map itself.

    UniDepth builds `points` as x = (u - cx) * z / fx (and likewise for y/v),
    so a least-squares fit of x/z against u recovers fx and cx exactly (the
    residual on this footage is ~1e-4 px). Doing it this way means the
    intrinsics are guaranteed consistent with the very points being lifted,
    and works on point maps cached before infer_unidepth_on_frames.py started
    saving intrinsics alongside them.
    """
    x, y, z = xyz
    height, width = xyz.shape[1:]
    valid = z > 1e-3
    u = np.broadcast_to(np.arange(width, dtype=np.float64), (height, width))[valid]
    v = np.broadcast_to(np.arange(height, dtype=np.float64)[:, None], (height, width))[valid]

    def fit(pixel, ratio):
        slope, intercept = np.linalg.lstsq(
            np.stack([pixel, np.ones_like(pixel)], axis=1), ratio, rcond=None)[0]
        return 1.0 / slope, -intercept / slope

    fx, cx = fit(u, (x / z)[valid])
    fy, cy = fit(v, (y / z)[valid])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _camera_intrinsics(xyz: np.ndarray, k_path: Path) -> np.ndarray:
    """UniDepth's own predicted intrinsics if cached and consistent with the
    point map, else the ones recovered from the point map."""
    recovered = _intrinsics_from_point_map(xyz)
    if not k_path.exists():
        return recovered
    saved = np.load(k_path).reshape(3, 3)
    # Guards against a saved K that belongs to a different resolution than the
    # point map (UniDepth resizes internally), which would silently misplace
    # every box rather than fail.
    if np.allclose(saved, recovered, rtol=1e-2, atol=1e-2):
        return saved
    print(f"Warning: {k_path.name} disagrees with its point map "
          f"(fx {saved[0, 0]:.1f} vs {recovered[0, 0]:.1f}); using the recovered intrinsics.")
    return recovered


def _clip_intrinsics(xyz_dir: Path, sample: int = CLIP_INTRINSICS_SAMPLE):
    """One camera for the whole clip: the element-wise median of the intrinsics
    recovered from an evenly spaced sample of its point maps.

    Median rather than mean because the per-frame estimates are not noisy about
    a centre so much as occasionally wild -- a frame of mostly sky, or a near
    wall, and UniDepth's focal head goes somewhere else entirely. The mean
    follows those; the median ignores them.

    Returns None if there are no point maps, leaving the per-frame path alone.
    """
    paths = sorted(xyz_dir.glob("*.xyz.npy"))
    if not paths:
        return None
    if len(paths) > sample:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, sample).astype(int)]

    resolved = []
    for path in paths:
        try:
            xyz = np.load(path)
        except (OSError, ValueError):
            continue
        resolved.append(_camera_intrinsics(
            xyz, path.with_name(path.name.replace(".xyz.npy", ".K.npy"))))
    if not resolved:
        return None

    stack = np.stack(resolved)
    clip_k = np.median(stack, axis=0)
    fx = stack[:, 0, 0]
    print(f"Clip camera from {len(resolved)} frame(s): fx={clip_k[0, 0]:.0f} "
          f"fy={clip_k[1, 1]:.0f} cx={clip_k[0, 2]:.0f} cy={clip_k[1, 2]:.0f}")
    print(f"  per-frame fx spanned {fx.min():.0f}-{fx.max():.0f} "
          f"({fx.max() / max(fx.min(), 1e-6):.2f}x); that spread is what this removes.")
    # The number this cannot fix, said plainly: a clip-wide fx that is wrong is
    # still wrong, and it is the run-end CLASS_SIZE_PRIOR check that sees it.
    return clip_k


def _erode_mask(mask_array: np.ndarray) -> np.ndarray:
    """Shrinks the mask away from its boundary, where depth bleeds onto the
    background. No-op if the mask is too small to survive it."""
    rows, cols = np.nonzero(mask_array)
    side = min(cols.max() - cols.min(), rows.max() - rows.min()) + 1
    radius = int(round(MASK_EROSION_FRAC * side / 2))
    if radius < 1:
        return mask_array
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    eroded = cv2.erode(mask_array.astype(np.uint8), kernel).astype(bool)
    return eroded if eroded.sum() >= MIN_MASK_POINTS else mask_array


def _reject_depth_outliers(points_cam: np.ndarray) -> np.ndarray:
    """Keeps points within DEPTH_BAND_FRAC of the mask's median depth."""
    # Non-positive depth is not a point at all (nothing projects from behind
    # the pinhole); dropping it first also keeps it out of the median, whose
    # sign the band below is relative to.
    points_cam = points_cam[:, points_cam[2] > 1e-3]
    if points_cam.shape[1] == 0:
        return points_cam
    z = points_cam[2]
    median = np.median(z)
    return points_cam[:, np.abs(z - median) <= DEPTH_BAND_FRAC * median]


def _fit_box(points_ego: np.ndarray):
    """points_ego: (3, N) points in ego convention. Returns [x,y,z,l,w,h,heading].

    Fits the heading in the ground plane only, by minimum-area rectangle
    (rotating calipers) over the points' (x, y) projection, and takes the
    height straight from the z extent. A free 3D fit (e.g. PCA over all three
    axes) cannot be used here: its axes are an arbitrary 3D frame, so neither
    the axis0/1/2 -> length/width/height labelling nor `atan2(R[1,0], R[0,0])`
    as a heading is meaningful unless one axis happens to come out vertical,
    and with monocular depth the dominant axis is reliably the viewing ray
    rather than the vehicle (observed: every box's heading within a few
    degrees of its own line of sight).
    """
    ground_xy = np.ascontiguousarray(points_ego[:2].T, dtype=np.float32)
    corners = cv2.boxPoints(cv2.minAreaRect(ground_xy))
    # Take the heading off the longer side rather than trusting minAreaRect's
    # angle convention (which differs across OpenCV versions); this also makes
    # length >= width by construction.
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    longer = max(edges, key=np.linalg.norm)
    heading = float(np.arctan2(longer[1], longer[0]))

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, sin_h], [-sin_h, cos_h]])  # ground -> box axes
    local = rotation @ points_ego[:2]
    low = np.percentile(local, EXTENT_PERCENTILE, axis=1)
    high = np.percentile(local, 100.0 - EXTENT_PERCENTILE, axis=1)
    z_low, z_high = np.percentile(points_ego[2], [EXTENT_PERCENTILE, 100.0 - EXTENT_PERCENTILE])

    center_xy = rotation.T @ ((low + high) / 2.0)
    return [float(center_xy[0]), float(center_xy[1]), float((z_low + z_high) / 2.0),
            float(high[0] - low[0]), float(high[1] - low[1]), float(z_high - z_low),
            heading]


def _box_corners_ego(box) -> np.ndarray:
    """(8, 3) ego-frame corners of a [x, y, z, l, w, h, heading] box."""
    x, y, z, length, width, height, yaw = box
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    local = signs * np.array([length, width, height]) / 2.0
    cos_h, sin_h = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]])
    return (rotation @ local.T).T + np.array([x, y, z])


def _projected_bbox(box, intrinsics: np.ndarray):
    """Pixel bbox (u0, v0, u1, v1) of a box's projected silhouette, or None if
    any corner is at or behind the camera (the projection is then unbounded)."""
    corners_ego = _box_corners_ego(box)
    transform = _ego_to_camera_transform()          # ego -> camera
    corners_cam = transform[:3, :3] @ corners_ego.T + transform[:3, 3:4]
    if np.any(corners_cam[2] <= 1e-3):
        return None
    uv = (intrinsics @ corners_cam)[:2] / corners_cam[2]
    return (uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max())


def _silhouette_grid(box, intrinsics: np.ndarray, roi, grid: int):
    """Rasterises a box's projected silhouette onto a `grid` x `grid` bitmap
    covering `roi` (u0, v0, u1, v1) in pixels. None if the box straddles the
    camera plane, where the projection is unbounded."""
    corners_ego = _box_corners_ego(box)
    transform = _ego_to_camera_transform()
    corners_cam = transform[:3, :3] @ corners_ego.T + transform[:3, 3:4]
    if np.any(corners_cam[2] <= 1e-3):
        return None
    uv = ((intrinsics @ corners_cam)[:2] / corners_cam[2]).T
    scale = np.array([grid / max(roi[2] - roi[0], 1e-6), grid / max(roi[3] - roi[1], 1e-6)])
    local = (uv - np.array([roi[0], roi[1]])) * scale
    canvas = np.zeros((grid, grid), np.uint8)
    hull = cv2.convexHull(np.clip(local, -1e4, 1e4).astype(np.float32))
    cv2.fillConvexPoly(canvas, hull.astype(np.int32), 1)
    return canvas.astype(bool)


def _face_away_from_camera(heading: float, front_ego: np.ndarray) -> float:
    """The representative of `heading` (defined only up to +-pi) that points away
    from the camera.

    Both representatives describe the same box, so this is purely about which way
    the vehicle is taken to face. Pointing away is the safer default: the wrong
    one puts the box's far end nearer than the mask it is anchored to, which
    projects larger and flares into a wedge across the frame.
    """
    view = front_ego[:2] / max(np.linalg.norm(front_ego[:2]), 1e-9)
    if np.cos(heading) * view[0] + np.sin(heading) * view[1] < 0.0:
        return float(np.arctan2(-np.sin(heading), -np.cos(heading)))
    return float(heading)


def _free_edges(target, shape):
    """(horizontal, vertical) -- whether each of the mask bbox's extents is a
    real measurement of the object rather than a cut made by the frame border."""
    height, width = shape
    return (target[0] > TRUNCATION_MARGIN_PX and target[2] < width - 1 - TRUNCATION_MARGIN_PX,
            target[1] > TRUNCATION_MARGIN_PX and target[3] < height - 1 - TRUNCATION_MARGIN_PX)


def _alignment_shift(target, projected, free, shape):
    """Pixel shift bringing a projected silhouette onto the mask.

    On an axis the frame did not cut, the two are centred on each other. On a
    truncated axis they are aligned by the surviving edge instead, so the box
    keeps running off-screen on the side where the object does.
    """
    height, width = shape
    shift = []
    for axis, (low, high, limit) in enumerate(((target[0], target[2], width),
                                               (target[1], target[3], height))):
        low_cut = low <= TRUNCATION_MARGIN_PX
        high_cut = high >= limit - 1 - TRUNCATION_MARGIN_PX
        if free[axis] or (low_cut and high_cut):
            shift.append((low + high - projected[axis] - projected[axis + 2]) / 2.0)
        elif low_cut:                       # cut on the near side: align the far edges
            shift.append(high - projected[axis + 2])
        else:
            shift.append(low - projected[axis])
    return shift


def _align_to_mask(box, target, free, shape, depth: float, intrinsics: np.ndarray,
                   passes: int = 4) -> list:
    """Translates `box` sideways/vertically until its silhouette sits on the mask.

    Iterated rather than applied once: the pixel shift is converted to a metric
    translation at the box's near-surface depth, but the corner that sets the
    silhouette's edge generally sits further back, so it moves by less than the
    shift asked for. Each pass shrinks the residual (observed: one pass left a
    truncated truck's right edge 253 px inboard of its mask).
    """
    box = list(box)
    for _ in range(passes):
        projected = _projected_bbox(box, intrinsics)
        if projected is None:
            return box
        shift = _alignment_shift(target, projected, free, shape)
        if abs(shift[0]) < 0.5 and abs(shift[1]) < 0.5:
            break
        box[1] -= shift[0] * depth / intrinsics[0, 0]
        box[2] -= shift[1] * depth / intrinsics[1, 1]
    return box


def _scaled_to_mask(box, target, intrinsics: np.ndarray, free=(True, True),
                    passes: int = 3) -> list:
    """Uniformly scales `box`'s extent so its silhouette spans `target`'s bbox.

    Uniform, so the class prior's proportions -- and therefore the silhouette's
    shape at a given yaw, which is what the heading search compares -- survive
    the scaling. Iterated because a box's depth extent scales with it, which
    makes the projection mildly non-linear in the scale factor.

    Only extents the frame did not cut are used to set the scale: a truncated
    one is a measurement of the frame border, not of the object, and spanning it
    inflates the box (see TRUNCATION_MARGIN_PX). If both are cut there is
    nothing better to go on, so both are used.
    """
    box = list(box)
    for _ in range(passes):
        projected = _projected_bbox(box, intrinsics)
        if projected is None:
            return box
        ratios = [(target[2] - target[0]) / max(projected[2] - projected[0], 1e-6),
                  (target[3] - target[1]) / max(projected[3] - projected[1], 1e-6)]
        usable = [r for r, keep in zip(ratios, free) if keep] or ratios
        scale = max(usable)
        box[3] *= scale
        box[4] *= scale
        box[5] *= scale
    return box


def _fit_heading_to_mask(mask_array: np.ndarray, front_ego: np.ndarray, target,
                         intrinsics: np.ndarray, dims, free, depth: float) -> float:
    """Yaw whose projected silhouette best matches `mask_array`, by IoU.

    `dims` are the class prior's (length, width, height) -- used for their
    proportions only, since _scaled_to_mask refits the absolute size for every
    candidate. Ties (a symmetric silhouette, e.g. a car seen exactly head-on)
    resolve to whichever candidate came first, which is harmless: the two yaws
    that tie describe the same box.
    """
    patch = mask_array[int(target[1]):int(target[3]) + 1, int(target[0]):int(target[2]) + 1]
    if patch.size == 0:
        return 0.0
    reference = cv2.resize(patch.astype(np.uint8), (SILHOUETTE_GRID, SILHOUETTE_GRID),
                           interpolation=cv2.INTER_AREA).astype(bool)
    if not reference.any():
        return 0.0

    best_yaw, best_score = 0.0, -1.0
    for step in range(HEADING_SEARCH_STEPS):
        yaw = np.pi * step / HEADING_SEARCH_STEPS - np.pi / 2.0
        centre = front_ego + np.array([np.cos(yaw), np.sin(yaw), 0.0]) * dims[0] / 2.0
        candidate = _scaled_to_mask(
            [centre[0], centre[1], front_ego[2], dims[0], dims[1], dims[2], yaw],
            target, intrinsics, free)
        # Align before scoring, or the comparison is about position rather than
        # shape: rotating the box swings its centre by up to length/2 sideways
        # (~255 px at 18 m), which walks the silhouette clean off the mask and
        # scores 0 for every yaw except the few that happen to land on it.
        candidate = _align_to_mask(candidate, target, free, mask_array.shape,
                                   depth, intrinsics)
        silhouette = _silhouette_grid(candidate, intrinsics, target, SILHOUETTE_GRID)
        if silhouette is None:
            continue
        union = np.count_nonzero(silhouette | reference)
        iou = np.count_nonzero(silhouette & reference) / union if union else 0.0
        # cos(2*yaw), not cos(yaw): the heading is defined only up to +-pi, so the
        # prior must treat yaw and yaw+pi as equally parallel.
        score = iou * (1.0 - HEADING_PRIOR_WEIGHT * (1.0 - np.cos(2.0 * yaw)) / 2.0)
        if score > best_score:
            best_score, best_yaw = score, yaw
    return best_yaw


def _mask_coverage(box, reference: np.ndarray, roi, intrinsics: np.ndarray) -> float:
    """Fraction of the mask's pixels that fall inside the box's silhouette."""
    silhouette = _silhouette_grid(box, intrinsics, roi, SILHOUETTE_GRID)
    if silhouette is None:
        return 0.0
    total = np.count_nonzero(reference)
    return np.count_nonzero(silhouette & reference) / total if total else 0.0


def _ground_contact_depth(mask_array: np.ndarray, intrinsics: np.ndarray,
                          name: str) -> float:
    """Camera-frame depth of where this mask meets the road, or None.

    Returns None rather than a fallback so the caller can report how often the
    constraint spoke; every rejection below is a case where the mask's lowest
    row is not the object standing on the road.
    """
    rows, cols = np.nonzero(mask_array)
    bottom = int(rows.max())
    # A mask cut off by the bottom of the frame continues below it, so its
    # lowest visible row is a property of the frame border, not of the object.
    if bottom >= mask_array.shape[0] - 1 - TRUNCATION_MARGIN_PX:
        return None
    band = cols[rows >= bottom - CONTACT_BAND_PX]
    contact = np.array([float(np.median(band)), float(bottom), 1.0])

    ray_cam = np.linalg.inv(intrinsics) @ contact
    ray_ego = _CALIBRATION.cam_to_ego() @ ray_cam
    # The ray has to be going down to meet the road at all. The floor on how
    # steeply is GROUND_MAX_RANGE restated: t = height / -ray_ego[2].
    if ray_ego[2] >= -_CALIBRATION.height / GROUND_MAX_RANGE:
        return None
    # ray_cam's z is 1 by construction, so the ray parameter is the depth.
    depth = float(_CALIBRATION.height / -ray_ego[2])

    prior = CLASS_SIZE_PRIOR.get(name)
    if prior is not None:
        implied = (rows.max() - rows.min() + 1) * depth / intrinsics[1, 1]
        if not GROUND_SIZE_BAND[0] <= implied / prior[1] <= GROUND_SIZE_BAND[1]:
            return None
    return depth


def _anchor_box_to_mask(box, mask_array: np.ndarray, points_cam: np.ndarray,
                        intrinsics: np.ndarray, name: str) -> tuple:
    """Rebuilds `box` around its own 2D detection.

    Everything except the class proportions comes from the mask: the heading
    from the shape of its silhouette (see HEADING_FROM_MASK), the position from
    the ray through its centre at the object's range, and the size from fitting
    that silhouette to it. Nothing is taken from the point-cloud fit, which gets
    all three wrong for reasons the point cloud itself cannot reveal (see
    ANCHOR_BOXES_TO_MASK, HEADING_COLLAPSE_DEG).

    The range is the one number the mask cannot supply on its own, and where it
    comes from is the difference between a box that sits still and one that
    breathes: the road first, the point map only as a fallback. See
    GROUND_CONTACT_RANGE.

    The box is then shrunk as far as MIN_MASK_COVERAGE allows. Fitting to the
    mask's *bounding box* would keep an oriented box permanently oversized,
    since the bounding box of a hexagonal silhouette is much bigger than the
    hexagon; measuring against the mask's own pixels instead lets an oriented
    box sit snugly inside its detection.

    :returns: (box, whether its range came from the road)
    """
    rows, cols = np.nonzero(mask_array)
    target = (float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max()))
    # Range from the road where the road can supply it, and from the point map
    # where it cannot -- see GROUND_CONTACT_RANGE. The contact point sits under
    # the object's nearest visible edge, which is the same surface
    # NEAR_SURFACE_PERCENTILE is reaching for, so the two are interchangeable
    # here and everything downstream of `depth` is unchanged.
    depth = _ground_contact_depth(mask_array, intrinsics, name) \
        if GROUND_CONTACT_RANGE else None
    grounded = depth is not None
    if depth is None:
        depth = float(np.percentile(points_cam[2], NEAR_SURFACE_PERCENTILE))
    if not np.isfinite(depth) or depth <= 1e-3:
        return box, False
    free = _free_edges(target, mask_array.shape)

    # Front-face centre: the ray through the mask bbox centre, at the near depth.
    centre_pixel = np.array([(target[0] + target[2]) / 2.0,
                             (target[1] + target[3]) / 2.0, 1.0])
    front_cam = (np.linalg.inv(intrinsics) @ centre_pixel) * depth
    front_ego = _camera_to_ego(front_cam.reshape(3, 1)).reshape(3)

    # Proportions from the class prior; absolute size is refit below, so only
    # the ratios between these three matter.
    length = CLASS_LENGTH_PRIOR.get(name, max(float(box[3]), 1.0))
    prior = CLASS_SIZE_PRIOR.get(name, (length / 2.5, length / 3.0))
    dims = (length, prior[0], prior[1])

    if HEADING_FROM_MASK:
        heading = _fit_heading_to_mask(mask_array, front_ego, target, intrinsics,
                                       dims, free, depth)
        # The search spans a half turn, because a cuboid's silhouette is identical
        # under yaw and yaw+pi -- so it cannot tell a vehicle's nose from its tail.
        # Resolve it to the representative pointing away from the camera: on a
        # dashcam most tracked vehicles are receding, and this at least makes the
        # rendered heading arrow consistent rather than flipping frame to frame.
        # Oncoming traffic is drawn reversed; only motion can settle that.
        heading = _face_away_from_camera(heading, front_ego)
    else:
        # _fit_box reads the heading off the longer minAreaRect edge, so it is
        # only defined up to +-pi. Pick the representative pointing away from the
        # camera: with the other one the box extends from its front face
        # *towards* the lens, putting its far end nearer than the mask it is
        # anchored to, which projects larger and flares across the frame.
        heading = _face_away_from_camera(float(box[6]), front_ego)

    centre = front_ego + np.array([np.cos(heading), np.sin(heading), 0.0]) * dims[0] / 2.0
    anchored = _scaled_to_mask(
        [float(centre[0]), float(centre[1]), float(front_ego[2]),
         float(dims[0]), float(dims[1]), float(dims[2]), float(heading)],
        target, intrinsics, free)

    # The construction centred the box on the mask bbox's centre ray, which is
    # the wrong anchor for a truncated mask -- align it before anything else, so
    # a box that never survives a shrink step is still placed correctly.
    anchored = _align_to_mask(anchored, target, free, mask_array.shape, depth, intrinsics)

    patch = mask_array[int(target[1]):int(target[3]) + 1, int(target[0]):int(target[2]) + 1]
    if patch.size == 0:
        return anchored, grounded
    reference = cv2.resize(patch.astype(np.uint8), (SILHOUETTE_GRID, SILHOUETTE_GRID),
                           interpolation=cv2.INTER_AREA).astype(bool)

    # _scaled_to_mask sizes the box so its silhouette's bounding box contains the
    # mask's; walk that back while the silhouette still covers the mask itself.
    for _ in range(COVER_ITERATIONS):
        trial = list(anchored)
        trial[3] *= SHRINK_STEP
        trial[4] *= SHRINK_STEP
        trial[5] *= SHRINK_STEP
        # Realign on the mask each step: shrinking about the box centre pulls the
        # silhouette off the detection otherwise. The shift is measured in the
        # image and converted to a camera-frame translation at the box's own
        # depth, then into ego axes (_camera_to_ego maps a camera delta
        # (dx, dy, 0) to an ego delta (0, -dx, -dy)).
        trial = _align_to_mask(trial, target, free, mask_array.shape, depth, intrinsics)
        if _mask_coverage(trial, reference, target, intrinsics) < MIN_MASK_COVERAGE:
            break
        anchored = trial
    return anchored, grounded


def _implied_extent(mask_array: np.ndarray, points_cam: np.ndarray,
                    intrinsics: np.ndarray) -> tuple:
    """(width, height) in metres that this mask's pixel bbox, its median depth
    and `intrinsics` together imply for the object -- the quantity compared
    against CLASS_SIZE_PRIOR.

    Deliberately measured from the mask bbox rather than from the fitted box:
    the fit is downstream of the depth band, the erosion and the percentile
    trim, any of which could be blamed for a small box. The bbox is the raw
    detection, so a disagreement here can only come from the depth or the
    intrinsics.
    """
    rows, cols = np.nonzero(mask_array)
    depth = float(np.median(points_cam[2]))
    width_px = float(cols.max() - cols.min() + 1)
    height_px = float(rows.max() - rows.min() + 1)
    return (width_px * depth / intrinsics[0, 0],
            height_px * depth / intrinsics[1, 1])


def _heading_offset_deg(box: list) -> float:
    """Angle between a box's heading and its own bearing from the camera, in
    degrees, folded to [0, 90] -- the heading is only defined up to +-pi (it is
    read off the longer minAreaRect edge), so 180 deg apart is the same axis."""
    bearing = np.arctan2(box[1], box[0])
    return float(abs((np.degrees(box[6] - bearing) + 90.0) % 180.0 - 90.0))


def _report_sanity(records: list) -> None:
    """Prints the size/heading sanity report for a completed run.

    Aggregated over the whole run rather than warned per detection: a single
    object can legitimately be off (odd pose, truncated mask, merged detection),
    so only the median over many carries any signal about the camera model.
    """
    if not records:
        return
    names = np.array([r[0] for r in records])
    ratios = np.array([r[1] for r in records])          # (N, 2) width, height
    offsets = np.array([r[2] for r in records])
    retention = np.array([r[3] for r in records])

    print("\nGeometry sanity check -- implied object size (mask bbox + median depth "
          "+ intrinsics) vs CLASS_SIZE_PRIOR:")
    reported = np.zeros(len(records), dtype=bool)
    for name in sorted(set(names)):
        rows = ratios[names == name]
        if len(rows) < MIN_SANITY_SAMPLES:
            continue
        reported |= names == name
        width_ratio, height_ratio = np.median(rows, axis=0)
        flag = "" if all(SIZE_RATIO_BOUNDS[0] <= r <= SIZE_RATIO_BOUNDS[1]
                         for r in (width_ratio, height_ratio)) else "   <-- off"
        print(f"  {name:12s} {len(rows):5d} dets   width {width_ratio:5.2f}x prior   "
              f"height {height_ratio:5.2f}x prior{flag}")
    if not reported.any():
        print(f"  (no class reached {MIN_SANITY_SAMPLES} detections with a known prior)")
        return

    collapsed = float((offsets < HEADING_COLLAPSE_DEG).mean())
    print(f"  heading collapse: {collapsed:.0%} of boxes point within "
          f"{HEADING_COLLAPSE_DEG:.0f} deg of their own line of sight "
          f"(~11% expected by chance; median offset {np.median(offsets):.1f} deg)")

    # Retention separates the two ways a box ends up too short. A low *implied*
    # size with retention near 1 means the camera model shrank everything before
    # fitting; retention well below 1 means _reject_depth_outliers' band sliced a
    # stripe out of the object, which the implied size cannot see because it is
    # computed from the mask bbox rather than from the surviving points.
    sliced = float((retention < 0.5).mean())
    print(f"  height retention (raw fit / mask-implied): median "
          f"{np.median(retention):.0%}; {sliced:.0%} of boxes kept under half "
          f"(depth band slicing the object, not the camera model)")

    # Pooled over detections, not over class medians: the ratios are already
    # normalised by each class's own prior, and a class seen 30 times must not
    # weigh the same as one seen 1300 times.
    scale = float(np.median(ratios[reported]))
    if not SIZE_RATIO_BOUNDS[0] <= scale <= SIZE_RATIO_BOUNDS[1]:
        print(f"\nWARNING: lifted objects are consistently {1.0 / scale:.1f}x "
              f"{'too small' if scale < 1 else 'too large'}. The usual cause is the "
              "focal length: an fx that is Nx too large shrinks every "
              "back-projected cross-section by N while leaving depth plausible, and "
              "no other check in this pipeline can see it (see CLASS_SIZE_PRIOR). "
              f"Point maps here imply a horizontal FOV that would need to be "
              f"{'wider' if scale < 1 else 'narrower'} to match these detections; "
              "pass the true intrinsics to infer_unidepth_on_frames.py rather than "
              "letting UniDepth predict them, and undistort wide-angle footage first.")
    if collapsed > 0.5:
        print(f"\nWARNING: {collapsed:.0%} of point-cloud fits have collapsed onto the "
              "line of sight -- rods pointed at the camera rather than fitted vehicles "
              "(see HEADING_COLLAPSE_DEG), their length the depth smear and not the "
              "object. This follows from the size error above whenever the cross-section "
              "is shrunk below the depth noise.")
        # Said explicitly because the number above is measured on the raw fit, and
        # with ANCHOR_BOXES_TO_MASK on that fit is discarded: the emitted boxes take
        # their proportions from CLASS_LENGTH_PRIOR/CLASS_SIZE_PRIOR and their heading
        # from the mask silhouette. Read as a statement about the *output* -- which is
        # how it reads -- this would send you looking for a fitting bug that the
        # anchoring already works around, when the only thing still wrong is the scale
        # the silhouette is fitted through, i.e. the focal length.
        if ANCHOR_BOXES_TO_MASK:
            print("  Note: ANCHOR_BOXES_TO_MASK is on, so these fits are not what was "
                  "written out --\n  every emitted box is already prior-proportioned and "
                  "mask-headed. What survives\n  into the output is the absolute size, "
                  "which _scaled_to_mask fits through the\n  intrinsics above; that is "
                  "the same one number as the size error.")


def _frame_image(frames_dir: Path, frame: str, cache: dict, shape) -> np.ndarray:
    """The original frame as BGR, resized to `shape` (the point map's
    resolution, which is what the masks were resized to) so it can be indexed
    with a mask.

    Only needed for traffic lights, whose lit state has to be read off the
    pixels; the cache holds a single frame because masks_json comes out of
    infer_frames.py grouped by frame, so nothing older is ever asked for again.
    """
    if frames_dir is None:
        raise ValueError(
            "Traffic lights were detected, but --frames_dir was not given: their lit "
            "state (red/yellow/green) can only be read off the original frames.")
    if frame not in cache:
        image = cv2.imread(str(frames_dir / frame))
        if image is None:
            raise FileNotFoundError(
                f"{frames_dir / frame} not found; it is the frame a traffic-light "
                "detection came from, so its state cannot be read.")
        if image.shape[:2] != tuple(shape):
            image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        cache.clear()
        cache[frame] = image
    return cache[frame]


def _traffic_light_position(mask_array: np.ndarray, intrinsics: np.ndarray) -> list:
    """Returns the bottom-face centre (the position renderer.draw_cuboid_at
    places a cuboid by) of the traffic light covered by `mask_array`, in ego
    coordinates.

    The direction comes from the mask centroid's viewing ray, so the cuboid is
    centred on the detection in the image whatever its range turns out to be.
    Doing it the other way round -- taking the centroid of the lifted points
    directly -- would drag the cuboid off its own light whenever the mask's few
    dozen pixels straddle a depth discontinuity (a light against the sky always
    does), because the surviving points are then off-centre in the mask.

    The range along that ray is the head's apparent size against the fixed size
    the renderer will draw it at, and pointedly *not* the point map. Every
    reason GROUND_CONTACT_RANGE gives for distrusting monocular depth applies
    here twice over, because a traffic light hangs against the sky, which is
    where the depth map has nothing to work with: over the ambulance clip
    UniDepth put the whole sky at 9-12 m and never returned anything beyond
    38 m, so its lights came back at a median 8 m when they are 25-50 m out.
    A cuboid at an eighth of its true range is drawn eight times too big, which
    is the wall of yellow those frames render as.

    Ground contact, the route vehicles take, is not available to something in
    the air -- so this falls to the size-based fit, the middle row of
    GROUND_CONTACT_RANGE's jitter table (0.073 against the point map's 0.087).
    Sizing off the drawn cuboid's own height rather than a separate real-world
    prior makes the result self-consistent: the cuboid lands on the image at
    exactly the extent of the mask it came from, whatever the clip's intrinsics
    are, so an fx error that would move a metric range cancels here.

    Measured along the head's longer axis and against TRAFFIC_LIGHT_DIMS's
    height, since that is the three-lens run whichever way the head is hung; a
    horizontal head still gets its range from the axis the lenses lie along,
    even though the cuboid is then drawn upright over a wide light.
    """
    rows, cols = np.nonzero(mask_array)
    height = float(rows.max() - rows.min()) + 1.0
    width = float(cols.max() - cols.min()) + 1.0
    # fy with rows, fx with columns: the two are not equal on the intrinsics
    # UniDepth predicts (2024 vs 2080 on this clip).
    long_axis, focal = ((height, intrinsics[1, 1]) if height >= width
                        else (width, intrinsics[0, 0]))
    depth = focal * TRAFFIC_LIGHT_DIMS[2] / long_axis

    pixel = np.array([cols.mean(), rows.mean(), 1.0])
    # Ray through the centroid, normalised so its z is 1: scaling it by a
    # depth therefore lands exactly at that depth.
    ray_cam = np.linalg.inv(intrinsics) @ pixel
    centre_cam = ray_cam * depth

    centre_ego = _camera_to_ego(centre_cam.reshape(3, 1)).reshape(3)
    centre_ego[2] -= TRAFFIC_LIGHT_DIMS[2] / 2.0  # centre -> bottom face
    return centre_ego.tolist()


def _point_map(xyz_dir: Path, frame: str, cache: dict):
    """The frame's UniDepth point map, or None if it was never inferred.

    Cached one frame deep: masks_json is grouped by frame, so without this the
    same 25 MB array is re-read from NFS once per detection in the frame.
    """
    if frame not in cache:
        xyz_path = xyz_dir / Path(frame).with_suffix(".xyz.npy").name
        cache.clear()
        cache[frame] = np.load(xyz_path) if xyz_path.exists() else None
    return cache[frame]


def _frame_entry(boxes_by_frame: dict, frame: str, xyz: np.ndarray, xyz_dir: Path,
                 intrinsics_cache: dict, clip_intrinsics=None) -> dict:
    """The frame's entry in the output, created (with its camera) on first use.

    Writing the clip camera into the per-frame cache rather than bypassing it
    keeps every later read -- _implied_extent, _anchor_box_to_mask, the traffic
    light lift, raster_frames -- on the one camera without any of them needing
    to know which mode this ran in.
    """
    if frame not in intrinsics_cache:
        xyz_path = xyz_dir / Path(frame).with_suffix(".xyz.npy").name
        intrinsics_cache[frame] = clip_intrinsics if clip_intrinsics is not None \
            else _camera_intrinsics(
                xyz, xyz_path.with_name(xyz_path.name.replace(".xyz.npy", ".K.npy")))
    return boxes_by_frame.setdefault(frame, {
        "boxes": [], "names": [], "scores": [], "manual_tracks": [],
        "traffic_lights": [], "lanes": [],
        "camera": {
            "intrinsics": intrinsics_cache[frame].tolist(),
            "ego_to_camera": _ego_to_camera_transform().tolist(),
            "image_hw": list(xyz.shape[1:]),
        },
    })


def _lift_lanes(boxes_by_frame: dict, frames_dir: Path, lane_masks_dir: Path,
                xyz_dir: Path, intrinsics_cache: dict, xyz_cache: dict,
                clip_intrinsics=None) -> int:
    """Adds each frame's lane polylines to its entry, creating entries for
    frames that had no object detections at all (a lane-only frame still has a
    map to draw)."""
    frame_paths = sorted(p for p in frames_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    lifted = 0
    for frame_path in tqdm(frame_paths, desc="lanes"):
        mask_path = lane_masks_dir / f"{frame_path.stem}.png"
        if not mask_path.exists():
            continue
        xyz = _point_map(xyz_dir, frame_path.name, xyz_cache)
        if xyz is None:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        if mask.shape != xyz.shape[1:]:
            mask = cv2.resize(mask.astype(np.uint8), (xyz.shape[2], xyz.shape[1]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue

        entry = _frame_entry(boxes_by_frame, frame_path.name, xyz, xyz_dir,
                             intrinsics_cache, clip_intrinsics)
        polylines = lane_polylines(mask, xyz, np.asarray(entry["camera"]["intrinsics"]),
                                   _CALIBRATION)
        entry["lanes"] = polylines
        lifted += len(polylines)

    # Identity across frames, once every frame has been lifted: a marking the
    # detector loses for a moment is held, and one it only ever glimpsed is
    # dropped. Per-frame gates cannot do either -- see lanes.stabilise_lanes.
    order = [p.name for p in frame_paths if p.name in boxes_by_frame]
    tracked, dropped, filled = stabilise_lanes(
        [boxes_by_frame[name]["lanes"] for name in order])
    for name, polylines in zip(order, tracked):
        boxes_by_frame[name]["lanes"] = [p.tolist() for p in polylines]
    # Frames that never got an entry keep whatever they had, which is nothing.
    for name, entry in boxes_by_frame.items():
        if name not in set(order):
            entry["lanes"] = [np.asarray(p).tolist() for p in entry.get("lanes", [])]
    print(f"Lane tracks: dropped {dropped} polyline(s) in tracks shorter than "
          f"{lanes_module.MIN_LANE_TRACK_LEN} frames, filled {filled} across dropouts.")
    return lifted


def _suppress_ego_masks(masks_json):
    """Drops detections of the car the camera is mounted on.

    Two stages, for the reasons set out at EGO_MASK_WIDTH_FRACTION. First the
    geometry -- a mask spanning most of the frame width and reaching its bottom
    edge -- which is necessary but not sufficient, because a vehicle that hits
    the ego gets merged into the bonnet's own mask. Then recurrence, which is
    what tells the two apart: the bonnet is bolted to the camera and comes back
    as the same pixels every time, a struck car does not.

    Runs before duplicate suppression so the bonnet cannot win a containment
    test against a real detection sitting on it.

    The first stage works off the RLE bounding box, which needs no allocation;
    only the survivors of it are paired up for the O(n^2) IoU, and those are a
    few per cent of a clip's detections.
    """
    candidates = []
    for index, detection in enumerate(masks_json):
        height, width = detection["mask"]["size"]
        _, top, box_width, box_height = mask_utils.toBbox(detection["mask"])
        if (box_width >= EGO_MASK_WIDTH_FRACTION * width
                and top + box_height >= height - EGO_MASK_BOTTOM_MARGIN_PX):
            candidates.append(index)
    if not candidates:
        return masks_json

    rles = [masks_json[i]["mask"] for i in candidates]
    overlaps = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
    np.fill_diagonal(overlaps, 0.0)          # a mask always matches itself
    recurrence = (overlaps >= EGO_MASK_RECURRENCE_IOU).sum(axis=1)
    ego = {index for index, count in zip(candidates, recurrence)
           if count >= EGO_MASK_MIN_RECURRENCE}

    kept = [d for i, d in enumerate(masks_json) if i not in ego]
    if ego:
        print(f"Suppressed {len(ego)} ego-vehicle masks (spanning over "
              f"{EGO_MASK_WIDTH_FRACTION:.0%} of the frame width, touching its bottom "
              f"edge, and recurring unchanged in at least {EGO_MASK_MIN_RECURRENCE} "
              f"other frames -- the camera's own bonnet).")
    if len(candidates) > len(ego):
        print(f"  ... and kept {len(candidates) - len(ego)} that matched the geometry "
              f"but not the recurrence: a bonnet-shaped mask that does not repeat is "
              f"something the ego has run into, merged with the bonnet.")
    return kept


def _suppress_surface_masks(masks_json):
    """Drops detections that are a piece of the scene rather than an object.

    Sprawling, flat and hollow all at once, and clear of the bottom edge, as set
    out at SURFACE_MASK_FILL -- the signature of the road surface being returned
    as a vehicle. Like _suppress_ego_masks this runs off the RLE bounding box and
    area, so it costs nothing per detection.
    """
    kept, dropped = [], 0
    for detection in masks_json:
        height, width = detection["mask"]["size"]
        _, top, box_width, box_height = mask_utils.toBbox(detection["mask"])
        area = float(mask_utils.area(detection["mask"]))
        if (area > SURFACE_MASK_AREA_FRACTION * width * height
                and box_width > SURFACE_MASK_ASPECT * max(box_height, 1.0)
                and area < SURFACE_MASK_FILL * max(box_width * box_height, 1.0)
                and top + box_height < (1.0 - SURFACE_MASK_BOTTOM_CLEARANCE) * height):
            dropped += 1
            continue
        kept.append(detection)
    if dropped:
        print(f"Suppressed {dropped} surface masks (over "
              f"{SURFACE_MASK_AREA_FRACTION:.0%} of the frame, over "
              f"{SURFACE_MASK_ASPECT}x wider than tall, and filling under "
              f"{SURFACE_MASK_FILL:.0%} of their own bounding box -- road, not vehicles).")
    return kept


def _suppress_duplicate_masks(masks_json):
    """Drops detections whose mask duplicates or nests inside a higher-scoring one.

    Two tests, because they catch different failures. IoU catches the detector
    emitting one mask twice under different labels; containment catches one mask
    swallowing another, where the intersection is large relative to the smaller
    mask but small relative to the union.

    Ties go to the higher score, which is the conventional NMS choice but not
    always the better mask: the leaky container often outscores the tight mask it
    contains (0.37 vs 0.35 in frame 66, at 59% vs 79% fill). Score does not
    predict mask quality on this footage, so there is no principled reordering
    available here -- fixing that means better masks, not better arbitration.
    """
    by_frame = {}
    for detection in masks_json:
        by_frame.setdefault(detection["frame"], []).append(detection)

    kept, dropped = [], 0
    for detections in by_frame.values():
        order = sorted(detections, key=lambda d: -d["score"])
        decoded, survivors = [], []
        for detection in order:
            mask_array = mask_utils.decode(detection["mask"]).astype(bool)
            area = np.count_nonzero(mask_array)
            duplicate = False
            for other in decoded:
                intersection = np.count_nonzero(mask_array & other)
                if not intersection:
                    continue
                union = np.count_nonzero(mask_array | other)
                smaller = min(area, np.count_nonzero(other))
                if (intersection / union > DUPLICATE_MASK_IOU
                        or intersection / smaller > DUPLICATE_MASK_CONTAINMENT):
                    duplicate = True
                    break
            if duplicate:
                dropped += 1
                continue
            decoded.append(mask_array)
            survivors.append(detection)
        kept.extend(survivors)
    if dropped:
        print(f"Suppressed {dropped} duplicate masks (IoU > {DUPLICATE_MASK_IOU} or "
              f"containment > {DUPLICATE_MASK_CONTAINMENT} with a higher-scoring "
              f"detection in the same frame).")
    return kept


def _contact_rows(masks_json, image_hw) -> list:
    """(top row, bottom row, class prior height) per detection, for the pitch
    refinement. Decoded here because this is where pycocotools already is."""
    rows = []
    for mask_dict in masks_json:
        if is_traffic_light(mask_dict["category"]):
            continue
        mask_array = mask_utils.decode(mask_dict["mask"]).astype(bool)
        if mask_array.shape != tuple(image_hw) or not mask_array.any():
            continue
        pixel_rows = np.nonzero(mask_array)[0]
        rows.append((int(pixel_rows.min()), int(pixel_rows.max()),
                     mask_dict["category"]))
    priors = {name: prior[1] for name, prior in CLASS_SIZE_PRIOR.items()}
    return calibrate_ground.contacts_from_masks(rows, image_hw, priors)


def _calibration_for_clip(xyz_dir: Path, calibration_masks_dir: Path,
                          clip_intrinsics, masks_json):
    """calibrate_ground's estimate for this clip, or the level default.

    Needs one point map only for its shape and, when CLIP_INTRINSICS is off, its
    camera -- the estimate itself never reads a depth, which is the whole reason
    it can be trusted about a clip whose depths are not rigid.
    """
    if calibration_masks_dir is None or not Path(calibration_masks_dir).is_dir():
        return calibrate_ground.GroundCalibration(height=CAMERA_HEIGHT,
                                                  source="no lane masks")
    paths = sorted(xyz_dir.glob("*.xyz.npy"))
    if not paths:
        return calibrate_ground.GroundCalibration(height=CAMERA_HEIGHT,
                                                  source="no point maps")
    xyz = np.load(paths[0])
    intrinsics = clip_intrinsics if clip_intrinsics is not None else \
        _camera_intrinsics(xyz, paths[0].with_name(
            paths[0].name.replace(".xyz.npy", ".K.npy")))
    return calibrate_ground.estimate(Path(calibration_masks_dir), intrinsics,
                                     xyz.shape[1:], CAMERA_HEIGHT,
                                     _contact_rows(masks_json, xyz.shape[1:]))


def lift_frames(masks_json, xyz_dir: Path, frames_dir: Path = None,
                lane_masks_dir: Path = None,
                calibration_masks_dir: Path = None) -> dict:
    masks_json = _suppress_duplicate_masks(
        _suppress_surface_masks(_suppress_ego_masks(masks_json)))
    boxes_by_frame = {}
    intrinsics_cache = {}
    frame_cache = {}
    xyz_cache = {}
    clip_intrinsics = _clip_intrinsics(xyz_dir) if CLIP_INTRINSICS else None
    # Before anything is lifted: every transform below reads it, and the boxes
    # written under one attitude cannot be reconciled with lanes written under
    # another.
    set_calibration(_calibration_for_clip(xyz_dir, calibration_masks_dir,
                                          clip_intrinsics, masks_json))
    print(f"Camera: {_CALIBRATION.describe()}")
    skipped = 0
    grounded = 0        # boxes whose range came from the road, not the point map
    anchored_total = 0
    sanity = []          # (class name, [width, height] ratio vs prior, heading offset)

    for mask_dict in tqdm(masks_json):
        frame = mask_dict["frame"]
        xyz = _point_map(xyz_dir, frame, xyz_cache)  # (3, H, W), camera frame
        if xyz is None:
            skipped += 1
            continue

        mask_array = mask_utils.decode(mask_dict["mask"]).astype(bool)  # (H, W)
        if mask_array.shape != xyz.shape[1:]:
            mask_array = cv2.resize(
                mask_array.astype(np.uint8), (xyz.shape[2], xyz.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if mask_array.sum() < MIN_MASK_POINTS:
            skipped += 1
            continue

        points_cam = xyz[:, _erode_mask(mask_array)]  # (3, N)
        points_cam = _reject_depth_outliers(points_cam)
        if points_cam.shape[1] < MIN_MASK_POINTS:
            skipped += 1
            continue

        entry = _frame_entry(boxes_by_frame, frame, xyz, xyz_dir,
                             intrinsics_cache, clip_intrinsics)

        if is_traffic_light(mask_dict["category"]):
            entry["traffic_lights"].append({
                "position": _traffic_light_position(
                    mask_array, intrinsics_cache[frame]),
                "state": classify_state(_frame_image(frames_dir, frame, frame_cache,
                                                     mask_array.shape), mask_array),
                "score": mask_dict["score"],
            })
        else:
            box = _fit_box(_camera_to_ego(points_cam))
            prior = CLASS_SIZE_PRIOR.get(mask_dict["category"])
            if prior is not None:
                implied = _implied_extent(mask_array, points_cam, intrinsics_cache[frame])
                sanity.append((mask_dict["category"],
                               [implied[0] / prior[0], implied[1] / prior[1]],
                               _heading_offset_deg(box),
                               # Retention: how much of the height the mask implies
                               # survived into the raw fit. Measured pre-anchoring,
                               # which would otherwise force it to ~1 and hide the
                               # depth-band slicing it exists to detect.
                               float(box[5]) / max(implied[1], 1e-6)))
            if ANCHOR_BOXES_TO_MASK:
                box, from_ground = _anchor_box_to_mask(
                    box, mask_array, points_cam, intrinsics_cache[frame],
                    mask_dict["category"])
                grounded += from_ground
                anchored_total += 1
            entry["boxes"].append(box)
            entry["names"].append(mask_dict["category"])
            entry["scores"].append(mask_dict["score"])
            # The annotator's track id, or None for a detector box. Identity is
            # a measurement for a hand-drawn box and a guess for a detected one,
            # so it travels with the box instead of being re-derived downstream
            # -- see smooth_boxes.associate().
            entry["manual_tracks"].append(mask_dict.get("track"))

    lights = sum(len(v["traffic_lights"]) for v in boxes_by_frame.values())
    total = sum(len(v["boxes"]) for v in boxes_by_frame.values())
    print(f"Lifted {total} boxes and "
          f"{lights} traffic lights across {len(boxes_by_frame)} frames "
          f"({skipped} masks skipped).")
    if anchored_total:
        share = 100.0 * grounded / anchored_total
        print(f"Range from the road for {grounded}/{anchored_total} boxes "
              f"({share:.0f}%); the rest fell back to the point map "
              f"(truncated, above the horizon, or failing GROUND_SIZE_BAND).")
    _report_sanity(sanity)

    if lane_masks_dir is not None:
        lanes = _lift_lanes(boxes_by_frame, frames_dir, lane_masks_dir, xyz_dir,
                            intrinsics_cache, xyz_cache, clip_intrinsics)
        print(f"Lifted {lanes} lane polylines across {len(boxes_by_frame)} frames.")
    return boxes_by_frame


def main():
    parser = argparse.ArgumentParser(
        description="Lift infer_frames.py's 2D mask detections into 3D boxes using cached UniDepth point maps.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing mask_results_preds.json and samples-pseudodepth/ "
                             "(from infer_frames.py and infer_unidepth_on_frames.py); boxes_3d.json is written here.")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Directory of original frame images (infer_frames.py's --frames_dir). "
                             "Required if the detections include traffic lights: their lit state is "
                             "read off the frame pixels. Also required with --lane_masks.")
    parser.add_argument("--min_score", type=float, default=0.0,
                        help="Skip detections scoring below this. GroundingDINO's own floor "
                             "is 0.30 and everything above it reaches here, including the "
                             "background clutter it grounds weakly -- hedges, distant "
                             "parking, street furniture. A per-clip setting, not a global "
                             "one: distant real traffic also scores low.")
    parser.add_argument("--no_calibrate", action="store_true",
                        help="Skip calibrate_ground and lift with a level camera, as "
                             "this stage did before the vanishing-point estimate "
                             "existed. For comparing a clip against its old output; "
                             "the level assumption is wrong by 7 deg on wrongway/4.")
    parser.add_argument("--no_ground_range", action="store_true",
                        help="Take every box's range from the point map, as this stage "
                             "did before GROUND_CONTACT_RANGE. The range then breathes "
                             "with the depth map's per-frame scale -- see the constant.")
    parser.add_argument("--lane_masks", action="store_true",
                        help="Also lift detect_lanes.py's lane_masks/ (under --output_dir) into "
                             "per-frame lane polylines.")
    args = parser.parse_args()

    if args.no_ground_range:
        global GROUND_CONTACT_RANGE
        GROUND_CONTACT_RANGE = False

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else None

    with open(output_dir / "mask_results_preds.json") as f:
        masks_json = json.load(f)
    xyz_dir = output_dir / "samples-pseudodepth"

    lane_masks_dir = None
    if args.lane_masks:
        if frames_dir is None:
            parser.error("--lane_masks needs --frames_dir (the frame list drives the lane pass).")
        lane_masks_dir = output_dir / "lane_masks"
        if not lane_masks_dir.is_dir():
            parser.error(f"{lane_masks_dir} does not exist; run detect_lanes.py first.")

    if args.min_score > 0:
        before = len(masks_json)
        masks_json = [m for m in masks_json
                      if m.get("manual") or float(m["score"]) >= args.min_score]
        # Hand-drawn boxes carry score 1.0 and pass anyway; the explicit exemption
        # is so that stays true if that convention ever changes.
        print(f"Score floor {args.min_score}: {before - len(masks_json)} of {before} "
              f"detection(s) skipped.")

    # Calibration reads lane_masks/ whether or not --lane_masks asked for the
    # lanes themselves to be lifted: the masks are an observation of the camera
    # first and a thing to draw second.
    calibration_masks_dir = output_dir / "lane_masks"
    if args.no_calibrate or not calibration_masks_dir.is_dir():
        calibration_masks_dir = None

    boxes_by_frame = lift_frames(masks_json, xyz_dir, frames_dir, lane_masks_dir,
                                 calibration_masks_dir)

    # The fit works in 7-value arrays; the file carries named fields. Converting
    # here rather than in lift_frames keeps the conversion in one place and off
    # the hot path -- see box_schema. Roll and pitch come out 0: the heading
    # search fits a yaw only, so saying so explicitly is honest, and a later
    # stage that does estimate them writes into fields that already exist.
    for entry in boxes_by_frame.values():
        entry["boxes"] = box_schema.to_dicts(entry["boxes"])

    save_path = output_dir / "boxes_3d.json"
    box_schema.dump(boxes_by_frame, save_path)
    print(f"Saved 3D boxes to {save_path}")


if __name__ == "__main__":
    main()
