"""
Turns detect_lanes.py's per-frame lane masks into 3D polylines for the
rasterized view.

The lifting deliberately does not use the per-pixel depth of the lane markings
themselves, the way lift_frames_to_3d.py lifts object masks. Measured on this
footage, UniDepth's road surface is not planar: lane pixels come back at
z = +0.47 m on average within 10 m of the camera and z = -0.3 m or so beyond
it, so a per-pixel lift produces lines that float off the road near the camera
and sink through it further out -- an obvious artefact in a 3D view, and a
wrong one in the exported geometry. Lane markings are painted *on* the road,
so the road surface is the right constraint: this intersects each polyline
vertex's viewing ray with a ground plane (see LANE_PLANE_FROM_DEPTH for which
one, and why it is no longer fitted to the depth).

Note the overlay is exact regardless of which plane is fitted: every point on
a viewing ray projects back to the pixel the ray came from, so the drawn line
always lands on the painted marking. The plane only decides where along the
ray the point sits, which affects the exported 3D coordinates and the
renderer's depth shading -- not the alignment.

That invariance is also why a bad plane is easy to miss, and why the guards
here act on the extension rather than on the drawn overlay. A plane is only
used if it actually explains the lane points (MIN_PLANE_INLIER_FRACTION),
falling back to flat ground when the frame's depth is too degenerate to define
a surface; each extended stretch stops where it stops receding to the horizon
(_keep_receding), and again where it leaves the frame (EXTEND_FRAME_MARGIN).
Without them, a frame whose depth had collapsed fitted planes tilted up to 41
degrees and offset by 26 m, and the extension followed one 6 m under the road
and 15000 px off the side of the image, or up over the horizon and out of the
top of the frame as a line ruled across the sky.

The ego frame's z = 0 is the road, and lane z is a height above it. That was
not true while lift_frames_to_3d._camera_to_ego was a bare axis remap -- the
frame inherited the mount's pitch, so the road sat at a tilt in these
coordinates and the plane had to be fitted to find it. calibrate_ground now
measures the pitch, so the tilt is out of the frame and the road is where the
name says it is.
"""
import cv2
import numpy as np

# Dashed markings arrive as a string of separate blobs. Closing with a tall,
# narrow kernel bridges the gaps along the direction a lane runs in the image
# (roughly vertical for a forward-facing camera) without merging neighbouring
# lanes, which are separated horizontally.
DASH_BRIDGE_KERNEL = (5, 41)  # (width, height) in pixels at 1080p

# Components smaller than this are noise -- kerb glare, wet patches, the odd
# stripe of a crosswalk clipped by the mask.
#
# These were 200 and 40 while this was the only gate a component had to pass,
# and a single frame is a bad place to make the call: the second half of
# wrongway/4 is worn and backlit, its lane mask falls from 30-65k pixels to
# 1-7k, and a threshold set high enough to reject glare there also rejects the
# markings. stabilise_lanes now tests every component against the frames around
# it, which is a test glare cannot pass, so this one can afford to let more
# through. Measured on wrongway/4, tracked in both cases:
#
#   200 / 40   empty frames 22, count changes 13%, worst dropout 7 frames
#    80 / 25   empty frames  4, count changes 17%, worst dropout 4 frames
#    40 / 20   empty frames  4, count changes 16%, worst dropout 4 frames
#
# Note the per-frame flicker gets *worse* as these come down (32% -> 38% of
# adjacent frames disagree, untracked) and the tracked result gets better
# anyway. Below 80 nothing more is recovered and only junk is admitted.
MIN_COMPONENT_PIXELS = 80
# ... and a component must span at least this many image rows to be a lane
# rather than a horizontal marking (stop bar, crosswalk stripe).
MIN_COMPONENT_ROWS = 25

# A lane divider runs *along* the road; a stop bar, a crosswalk stripe or the
# kerb of a side road runs across it. Tested on the lifted polyline rather than
# on the mask, because the image tells you nothing here: a nearby edge line is
# almost horizontal in the image purely because it is close, and rejecting on
# image aspect throws it away along with the stop bars.
MIN_ALONG_ROAD_RATIO = 0.7

# A NEGATIVE RESULT, recorded so it is not re-tried: YOLOPv2's drivable-area
# head does not separate a lane divider from the kerb beside it. detect_lanes.py
# writes that mask (it comes free with the forward pass this pipeline was
# already making), and gating components on it is the obvious fix for the kerb
# YOLOPv2 segments through the end of wrongway/4. It does not work, because the
# drivable head disagrees with the premise: that kerb component's overlap with
# the *undilated* drivable mask is 0.51, so the segmentation calls half of it
# road. Any threshold loose enough to keep real edge lines -- which are painted
# at the boundary and straddle it -- keeps this too. Applying it anyway cost 2
# more empty frames and 0.4 polylines per frame with nothing to show for them.
#
# The masks are still written. They are the right input for a road-membership
# test; this particular test is just not one.

# Half-width of the moving average applied to the centreline's column values.
# The mask edge is ragged at the pixel level and a marking's own width varies
# (dash ends, glare), which shows up as visible kinks once the polyline is
# drawn; lanes are smooth over any real curvature at this sampling rate.
SMOOTHING_HALF_WIDTH = 2

# How far a centreline may move between two sampled rows, in pixels, before the
# trace decides the component has branched rather than bent. A lane at its most
# oblique still moves smoothly; a jump to a different marking does not. Measured
# on wrongway/4, sweeping this against the tortuosity of the lifted result:
#
#   100 px   13% of components squiggly (see MAX_LANE_TORTUOSITY), 700 kept
#    40 px    9%                                                   660 kept
#    25 px   10%, and the median gets worse -- it is now truncating real lanes
MAX_TRACE_STEP_PX = 40.0

# Total lateral wander of a lifted polyline, as a fraction of the forward
# distance it covers. This is the squiggle itself, measured: a marking that
# drifts 10 m sideways over 30 m of road -- a sharp bend -- is 0.33, and a real
# one cannot double back at all. Anything past this is a trace that crossed
# between markings despite MAX_TRACE_STEP_PX, and is dropped rather than drawn.
#
# Tested on the lifted polyline rather than on the mask, for the same reason
# MIN_ALONG_ROAD_RATIO is: in the image a near marking is legitimately steep and
# a far one legitimately flat, and only on the ground are the two comparable.
MAX_LANE_TORTUOSITY = 0.7

# The polyline takes one vertex per this many image rows. Lanes are smooth, so
# this is about keeping the exported geometry small rather than about fidelity.
ROW_STEP = 8

# Plane fit. Inlier band is generous because it only has to separate the road
# surface from gross outliers (a lane pixel that landed on a car bumper).
# Whether the ground plane is fitted to the point map's lane pixels or taken as
# the calibrated road, z = 0.
#
# It was fitted because it had to be: before calibrate_ground the ego frame
# carried the mount's pitch, so the road was at an unknown tilt and only the
# depth could say what it was. Now that the pitch is measured, the fit is
# strictly the worse of the two, and for the usual reason -- it is downstream of
# a point map whose scale breathes. Measured on wrongway/4, per frame:
#
#   plane forward tilt a    mean +0.011, sd 0.026     -- flat, as it should be
#   plane lateral tilt b    mean +0.001, sd 0.140     -- and -0.704 on frame 310,
#                                                        a 35 deg bank of the road
#   plane height c          mean +0.45 m, sd 0.55 m   -- should be 0
#
# so the lifted markings floated a median 0.56 m above the road (p95 1.26 m) and
# a badly banked frame threw its extensions sideways. The forward tilt coming
# out flat is the calibration working; everything else is noise the fit had no
# way to reject, because a plane through breathing points is still a plane.
#
# What this does NOT change is the overlay: every point on a viewing ray
# projects back to the pixel it came from, so the drawn line lands on the
# painted marking under any plane at all. What it changes is where along the ray
# the point sits -- the exported geometry, the renderer's depth shading, and
# (the visible part) the direction _extend continues the line in.
LANE_PLANE_FROM_DEPTH = False

RANSAC_ITERATIONS = 200
RANSAC_INLIER_BAND = 0.15  # metres
MIN_PLANE_POINTS = 50

# Fraction of the lane points the fitted plane must actually explain before it
# is trusted. Lane markings are painted on one surface, so when the depth
# behind them is sound a single plane accounts for essentially all of them:
# measured 1.00 on beepbeep, with a median residual of 2-19 mm. A frame whose
# depth has collapsed has no surface to find, and the fit degenerates without
# failing -- on changelane it returned planes tilted up to 41 degrees, offset
# by as much as 26 m, explaining only 24-56% of the points.
#
# Those planes are invisible in the overlay (a ray-plane intersection
# reprojects to its own pixel whatever the plane is, see the module docstring)
# but they are what _extend follows, and following one put the near end of a
# lane 6 m under the road, drawn diving through the ego's own bonnet. Falling
# back to nominal flat ground is both safer and honest: it says the frame gave
# no usable surface rather than inventing a steeply banked one.
MIN_PLANE_INLIER_FRACTION = 0.8

# Beyond this the ray is so close to parallel with the road that a metre of
# plane error becomes tens of metres of range error, and the marking is a
# couple of pixels wide anyway.
MAX_RANGE = 60.0

# --- Extrapolation ---------------------------------------------------------
# A camera only sees paint where nothing is parked on top of it, so a detected
# marking stops at the first vehicle ahead. The road does not, and neither does
# the map-derived raster this is meant to match, so each lane is continued as a
# straight line to NEAR_RANGE..EXTEND_RANGE. Only the direction is extrapolated;
# the observed vertices are kept as measured.
EXTEND_RANGE = 60.0            # forward distance to continue lanes out to
NEAR_RANGE = 2.0               # ... and back towards the ego to
RESAMPLE_STEP = 1.0            # spacing of the extrapolated vertices, metres

# Fragments of one lane, split by whatever occluded it, are rejoined when their
# lateral offsets agree to within this. Too loose and neighbouring lanes merge
# into one; a lane is ~3.5 m from its neighbour, so this has plenty of margin.
MERGE_LATERAL_TOLERANCE = 0.7

# ... but lateral agreement alone does not say two fragments are one marking,
# because the markings that are close enough to agree are exactly the ones that
# run side by side: a double centre line, a lane line beside the shoulder edge,
# the two rails of a wide marking split down the middle by the mask. What
# separates those from a genuine occlusion is where they sit *along* the road.
# An occluder hides a stretch of paint, so the fragments it leaves lie end to
# end and hardly overlap in forward distance; two parallel markings overlap
# over the whole of the shorter one. So a merge is allowed only when the
# fragments share no more than this fraction of the shorter one's span.
#
# Merging a parallel pair does not merely place the line badly -- the merged
# fragments are sorted by forward distance, which interleaves the two rows of
# vertices, so the polyline zigzags from one marking to the other and back at
# every step. buick_nearmiss frame 31 joined two markings 0.5 m apart that
# overlapped over all 12 m of the shorter, and drew the pair as a 40-rung
# ladder across the road.
MAX_MERGE_OVERLAP_FRACTION = 0.25

# Direction is fitted to the whole fragment, except for fragments long enough
# that their far end is a better guide than a chord across the whole arc. Half
# of a short fragment is mostly noise: measured on this footage, fitting the
# far half of a ~10 m fragment steepened the extrapolated slope by 50% (-0.13
# to -0.20), moving the line 3 m sideways at 60 m for no gain in fidelity.
LONG_FRAGMENT_SPAN = 20.0
DIRECTION_FIT_FRACTION = 0.5
MIN_DIRECTION_POINTS = 3

# A fragment shorter than this says almost nothing about direction -- extending
# it to 60 m would be inventing geometry, so it is left as measured. The edge
# line in the bottom corner of a frame is the usual case: seen almost side-on
# over 2 m, it fits slopes around 0.7, which would fling the extension 40 m
# sideways.
MIN_SPAN_TO_EXTEND = 3.0

# How far a lane may be continued past its measured stretch, as a multiple of
# that stretch. The direction of an extension is fitted to the measured vertices
# and nothing else, so the error at the far end grows with how far it is carried
# and shrinks with how much road the fit had to work from -- a fraction of a
# degree over a few metres of near-field pixels is tens of metres of lateral
# error at 60 m. Reaching a fixed EXTEND_RANGE ignores both halves of that.
#
# Measured on wrongway/4 before this cap: the median polyline was measured over
# 7.2 m of road and drawn out to 56.3 m, 51% of them were extended past 3x their
# measured span and 38% past 6x, and 271 of 656 were measured over less than 6 m
# and still drawn to a median 55.8 m. That is what puts a line up the kerb and
# off across the pavement -- not a bad detection, a good one carried 10x too far.
MAX_EXTEND_SPAN_MULTIPLE = 1.0

# ... and even a long fragment is not extended along an implausible heading;
# 0.35 is a ~19 degree divergence from the ego's own direction, well beyond
# any lane the ego is driving along and into "this fit is wrong" territory.
MAX_EXTEND_SLOPE = 0.35

# How far outside the frame an extended vertex may still be trusted, as a
# multiple of the image size.
#
# The extension is the only geometry here with no ray behind it. A measured
# vertex is a ray-plane intersection, so it reprojects onto the pixel it came
# from whatever the plane is (see the module docstring) -- which means a badly
# fitted plane is invisible until _extend follows it somewhere the camera never
# looked. Measured on this footage: a frame whose plane came back as
# z = 0.170x - 6.485 (the road 6.5 m below the camera at the ego origin, tilted
# up 9.7 degrees) put the near end of a lane 7.6 m below a camera 2 m away, and
# the vertex landed 15709 px off the left edge -- drawn as a streak from there
# to the on-screen measured end.
#
# The bound has to be looser than the frame itself, because a lane really does
# leave it at close range: the ego's own markings pass out of the bottom edge.
# One full frame of overshoot separates the two cases cleanly -- clips whose
# depth supports a sane plane (beepbeep, redlight) produce no extended vertex
# beyond it at all, while the degenerate frames overshoot it by ten times.
EXTEND_FRAME_MARGIN = 1.0

# Slack, in pixels, on the "an extension must recede" test below. The vertices
# are samples of a straight line on a plane, so their image rows are exactly
# monotonic when the plane is sane; this only absorbs a vertex that barely
# moves at all.
EXTEND_ROW_TOLERANCE = 0.5


def _fit_ground_plane(points_ego: np.ndarray):
    """Fits z = a*x + b*y + c to (3, N) ego points by RANSAC, refined by least
    squares on the inliers. Returns (a, b, c), or None if there is not enough
    to fit -- the caller then falls back to the nominal z = 0 ground."""
    if points_ego.shape[1] < MIN_PLANE_POINTS:
        return None

    design = np.stack([points_ego[0], points_ego[1], np.ones(points_ego.shape[1])], axis=1)
    heights = points_ego[2]

    rng = np.random.default_rng(0)  # fixed seed: same frames must lift identically
    best_inliers = None
    for _ in range(RANSAC_ITERATIONS):
        sample = rng.choice(design.shape[0], size=3, replace=False)
        try:
            model = np.linalg.solve(design[sample], heights[sample])
        except np.linalg.LinAlgError:
            continue  # the three points were collinear in the ground plane
        inliers = np.abs(design @ model - heights) <= RANSAC_INLIER_BAND
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < MIN_PLANE_POINTS:
        return None
    model = np.linalg.lstsq(design[best_inliers], heights[best_inliers], rcond=None)[0]

    # Scored on every point rather than on RANSAC's own inliers, which would
    # just restate the sample the model was chosen from. A plane that leaves
    # most of the road unexplained is not a road (MIN_PLANE_INLIER_FRACTION).
    explained = np.abs(design @ model - heights) <= RANSAC_INLIER_BAND
    if explained.mean() < MIN_PLANE_INLIER_FRACTION:
        return None
    return model


def _row_runs(row: np.ndarray):
    """The maximal runs of set pixels in one row, as (first, last) columns."""
    columns = np.nonzero(row)[0]
    if not len(columns):
        return []
    breaks = np.nonzero(np.diff(columns) > 1)[0]
    return [(int(run[0]), int(run[-1])) for run in np.split(columns, breaks + 1)]


def _component_polyline(component: np.ndarray) -> np.ndarray:
    """Centreline of one connected component, as (N, 2) pixel coordinates
    ordered far-to-near (top of the image down).

    Traced one row at a time rather than taken as the mean column per row. The
    mean is the centreline only while the component is one thin marking, and
    nothing upstream guarantees that: YOLOPv2 segments crosswalk bars and kerbs
    along with the dividers, and where those touch, the component becomes a
    network. On frame 53 of wrongway/4 that gave a single component 999 px wide
    spanning 273 rows, 83% of whose rows held two separate runs -- and the mean
    of two runs is the empty tarmac between them, which is what drew a zigzag
    across the road.

    So the trace follows one branch: it starts at the nearest row, where the
    marking is best resolved, and at each row upward takes the run closest to
    where the line already is, stopping if the nearest one is further than
    MAX_TRACE_STEP_PX. Where a component really is a single marking this is the
    mean; where it is not, it is one of the markings instead of the average of
    all of them. Skeletonisation would be the general answer but needs ximgproc.
    """
    rows = np.nonzero(component.any(axis=1))[0][::ROW_STEP]
    if len(rows) < 2:
        return np.empty((0, 2))

    traced, centre = [], None
    for row in rows[::-1]:                      # nearest row first, then upward
        runs = _row_runs(component[row])
        if not runs:
            break
        if centre is None:
            # Seed on the widest run: at the near end of a marking that is the
            # marking, and on a network it is the piece with the most support.
            first, last = max(runs, key=lambda run: run[1] - run[0])
        else:
            first, last = min(runs, key=lambda run: abs((run[0] + run[1]) / 2 - centre))
            if abs((first + last) / 2 - centre) > MAX_TRACE_STEP_PX:
                break
        centre = (first + last) / 2.0
        traced.append((centre, float(row)))
    if len(traced) < 2:
        return np.empty((0, 2))

    columns = np.array([column for column, _ in traced])
    rows = np.array([row for _, row in traced])
    window = 2 * SMOOTHING_HALF_WIDTH + 1
    if len(columns) >= window:
        # 'edge' padding rather than zero padding, so the ends of the line stay
        # where they are instead of being pulled towards the image border.
        padded = np.pad(columns, SMOOTHING_HALF_WIDTH, mode="edge")
        columns = np.convolve(padded, np.ones(window) / window, mode="valid")
    # Back to far-to-near, which is the order every caller expects.
    return np.stack([columns, rows], axis=1)[::-1]


def _rays_to_ego(pixels: np.ndarray, intrinsics: np.ndarray, calibration):
    """Viewing rays through `pixels` ((N, 2) as x, y), as (origin, directions)
    in ego coordinates: origin (3,), directions (N, 3).

    The rotation is the direction-only part of lift_frames_to_3d._camera_to_ego
    (camera x-right/y-down/z-forward -> ego x-forward/y-left/z-up, plus whatever
    attitude calibrate_ground measured); the camera height becomes the ray
    origin instead of a translation on every point.
    """
    homogeneous = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
    directions_cam = homogeneous @ np.linalg.inv(intrinsics).T
    directions_ego = directions_cam @ calibration.cam_to_ego().T
    return np.array([0.0, 0.0, calibration.height]), directions_ego


def _intersect_ground(origin, directions, plane) -> np.ndarray:
    """Intersects each ray with z = a*x + b*y + c, returning (N, 3) ego points
    with the misses dropped: rays pointing at or above the horizon never meet
    the road, and those past MAX_RANGE are not worth keeping."""
    a, b, c = plane
    # Substituting origin + t*direction into z - (a*x + b*y + c) = 0.
    numerator = a * origin[0] + b * origin[1] + c - origin[2]
    denominator = directions[:, 2] - a * directions[:, 0] - b * directions[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerator / denominator
    points = origin + t[:, None] * directions
    valid = np.isfinite(t) & (t > 0) & (np.hypot(points[:, 0], points[:, 1]) <= MAX_RANGE)
    return points[valid]


def _project_ego(points: np.ndarray, intrinsics: np.ndarray, calibration) -> np.ndarray:
    """(N, 2) pixel coordinates of ego-frame `points` ((N, 3)).

    The inverse of the mapping _rays_to_ego builds its rays with: ego
    (x-forward, y-left, z-up) with the camera at the calibrated height and
    attitude, back to camera (x-right, y-down, z-forward).
    """
    offset = points - np.array([0.0, 0.0, calibration.height])
    camera = offset @ calibration.cam_to_ego()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (camera @ intrinsics.T)[:, :2] / camera[:, 2:3]


def _keep_inside_frame(points: np.ndarray, intrinsics: np.ndarray, calibration,
                       image_shape) -> np.ndarray:
    """Trims `points` -- ordered outward from the measured end -- at the first
    vertex that leaves the frame by more than EXTEND_FRAME_MARGIN.

    Truncated rather than filtered: the extension walks away from the measured
    data one step at a time, so once it has left the frame nothing further out
    is any more trustworthy, and keeping a later vertex that happens to land
    back inside would draw a segment across the whole image to reach it.
    """
    if len(points) == 0:
        return points
    height, width = image_shape
    pixels = _project_ego(points, intrinsics, calibration)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (points[:, 0] > 0)  # behind the camera has no projection to bound
        & (pixels[:, 0] >= -EXTEND_FRAME_MARGIN * width)
        & (pixels[:, 0] <= (1.0 + EXTEND_FRAME_MARGIN) * width)
        & (pixels[:, 1] >= -EXTEND_FRAME_MARGIN * height)
        & (pixels[:, 1] <= (1.0 + EXTEND_FRAME_MARGIN) * height)
    )
    outside = np.nonzero(~inside)[0]
    return points if len(outside) == 0 else points[:outside[0]]


def _keep_receding(points: np.ndarray, anchor: np.ndarray, at_far_end: bool,
                   intrinsics: np.ndarray, calibration) -> np.ndarray:
    """Trims `points` -- ordered outward from the measured end at `anchor` --
    at the first vertex that walks the wrong way up the image.

    Road surface recedes to a horizon: step away from the ego along a lane and
    each vertex projects higher in the frame (smaller row), step back towards
    the ego and each projects lower. That holds for any plane the camera is
    above and looking along, so an extension that breaks it is following a
    plane that is not a road.

    This is the one artefact EXTEND_FRAME_MARGIN cannot catch, because it stays
    comfortably inside the frame. Measured on changelane: a jam frame whose
    depth put the road at a 10 degree downward tilt sent the near extension of
    the ego's own lane *up* over the horizon -- 2 m in front of the car and
    drawn off the top edge, so the lane read as a line ruled across the sky --
    while the far extension of the same lane sank steadily below the measured
    end it grew out of. Both directions are wrong in the same way and both stop
    here, at the first vertex, leaving the measured part untouched.
    """
    if len(points) == 0:
        return points
    rows = _project_ego(np.concatenate([anchor[None], points]),
                        intrinsics, calibration)[:, 1]
    # Rows must fall along a near extension and rise along a far one.
    steps = np.diff(rows) * (-1.0 if at_far_end else 1.0)
    bad = np.nonzero(~(np.isfinite(steps) & (steps >= -EXTEND_ROW_TOLERANCE)))[0]
    return points if len(bad) == 0 else points[:bad[0]]


def _lateral_fit(points: np.ndarray, at_far_end: bool):
    """Fits y = m*x + b to one end of a lane, returning (m, b).

    Only the end being extended is fitted, so a curving lane is continued
    along its local tangent rather than along the chord of its whole arc.
    """
    ordered = points[np.argsort(points[:, 0])]
    if np.ptp(ordered[:, 0]) >= LONG_FRAGMENT_SPAN:
        take = max(MIN_DIRECTION_POINTS, int(round(DIRECTION_FIT_FRACTION * len(ordered))))
        ordered = ordered[-take:] if at_far_end else ordered[:take]
    if len(ordered) < 2 or np.ptp(ordered[:, 0]) < 1e-6:
        return None
    return np.polyfit(ordered[:, 0], ordered[:, 1], 1)


def _overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the shorter fragment's forward span that both cover.

    0 for fragments lying end to end along the road -- the shape an occluded
    marking leaves -- and 1 for one running alongside the other.
    """
    low = max(a[:, 0].min(), b[:, 0].min())
    high = min(a[:, 0].max(), b[:, 0].max())
    if high < low:
        return 0.0
    shorter = min(np.ptp(a[:, 0]), np.ptp(b[:, 0]))
    # A fragment with no span at all is a point; it overlaps entirely or not,
    # and the comparison above has already settled which.
    return 1.0 if shorter < 1e-6 else min(1.0, (high - low) / shorter)


def _merge_fragments(polylines: list) -> list:
    """Joins polylines that are pieces of the same marking.

    Two fragments belong together when the far end of one, extended, passes
    through the other at the other's own forward distances *and* the two do not
    cover the same stretch of road (MAX_MERGE_OVERLAP_FRACTION) -- which is
    exactly the situation a vehicle parked over the paint creates, and is not
    the situation two markings running side by side create.
    """
    remaining = sorted(polylines, key=lambda p: -np.ptp(p[:, 0]))  # longest first: best direction
    merged = []
    while remaining:
        lane = remaining.pop(0)
        absorbed = True
        while absorbed:  # a fragment joined on may reach fragments the original could not
            absorbed = False
            fit = _lateral_fit(lane, at_far_end=True)
            if fit is None:
                break
            # Rebuilt rather than removed from: `remaining` holds arrays, and
            # list.remove would compare them elementwise.
            unmatched = []
            for candidate in remaining:
                predicted = np.polyval(fit, candidate[:, 0])
                if np.median(np.abs(predicted - candidate[:, 1])) <= MERGE_LATERAL_TOLERANCE \
                        and _overlap_fraction(lane, candidate) <= MAX_MERGE_OVERLAP_FRACTION:
                    lane = np.concatenate([lane, candidate])
                    absorbed = True
                else:
                    unmatched.append(candidate)
            remaining = unmatched
            if absorbed:
                lane = lane[np.argsort(lane[:, 0])]
        merged.append(lane)
    return merged


def _extend(points: np.ndarray, plane, intrinsics: np.ndarray, calibration,
            image_shape) -> np.ndarray:
    """Continues a lane out towards EXTEND_RANGE and back towards NEAR_RANGE.

    The measured vertices are kept untouched in the middle; only the added
    stretches are model-generated. Heights come from the ground plane, so the
    extension stays on the same surface the lane was lifted onto -- and each
    stretch stops as soon as it stops behaving like a road, either by ceasing to
    recede towards the horizon (_keep_receding) or by leaving the frame
    (EXTEND_FRAME_MARGIN), so a badly fitted plane truncates the extension
    instead of flinging it across the image.
    """
    if np.ptp(points[:, 0]) < MIN_SPAN_TO_EXTEND:
        return points

    a, b, c = plane
    ordered = points[np.argsort(points[:, 0])]
    # What the fit has earned the right to say, in metres of road either way.
    reach = MAX_EXTEND_SPAN_MULTIPLE * np.ptp(points[:, 0])
    pieces = []
    for at_far_end in (False, True):
        fit = _lateral_fit(ordered, at_far_end)
        if fit is None or abs(fit[0]) > MAX_EXTEND_SLOPE:
            continue
        if at_far_end:
            start = ordered[-1, 0] + RESAMPLE_STEP
            stop = min(EXTEND_RANGE, ordered[-1, 0] + reach)
        else:
            start = max(NEAR_RANGE, ordered[0, 0] - reach)
            stop = ordered[0, 0] - RESAMPLE_STEP
        if stop < start:
            continue
        x = np.arange(start, stop + RESAMPLE_STEP / 2, RESAMPLE_STEP)
        y = np.polyval(fit, x)
        piece = np.stack([x, y, a * x + b * y + c], axis=1)
        # Both stretches are built in increasing x, but the near one grows
        # *away* from the measured vertices as x decreases; reverse it so
        # truncation always keeps the end that adjoins them.
        if not at_far_end:
            piece = piece[::-1]
        piece = _keep_receding(piece, ordered[-1] if at_far_end else ordered[0],
                               at_far_end, intrinsics, calibration)
        piece = _keep_inside_frame(piece, intrinsics, calibration, image_shape)
        if not at_far_end:
            piece = piece[::-1]
        if len(piece) == 0:
            continue
        pieces.append((piece, at_far_end))

    near = [p for p, far in pieces if not far]
    far = [p for p, far in pieces if far]
    return np.concatenate(near + [ordered] + far)


# --- holding a marking across the frames its detector loses it in -------------
#
# Everything above works on one frame and knows nothing about the last one, and
# on wrongway/4 that shows: the polyline count changes between 29% of adjacent
# frame pairs, 20 frames come back with no lanes at all, and one run of 7
# consecutive frames -- three quarters of a second -- is empty. The road did not
# move. YOLOPv2 lost it, mostly where the markings are worn or backlit: the lane
# mask falls from 30-65k pixels in the first half of the clip to 1-7k in the
# second, and MIN_COMPONENT_PIXELS then rejects what is left.
#
# So this is the same pass smooth_boxes.py runs over the boxes, on the same
# argument: identity across frames is what turns a detector's per-frame opinion
# into a thing that persists. A marking's lateral offset a fixed distance ahead
# is the quantity to associate on -- it barely moves under the ego's own forward
# motion, so no odometry is needed to compare one frame with the next -- and a
# track that survives is filled across the frames it is missing from.
#
# Dropping the short tracks is the other half, and it is what a per-frame gate
# cannot do: the kerb YOLOPv2 segments for a few frames near the end of
# wrongway/4 is a perfectly good lane-shaped component in every frame it appears
# in. What gives it away is that it is not there for long.

# Lateral tolerance for calling two polylines the same marking, in metres. A
# lane is 3.5 m wide, so anything under half of that cannot confuse neighbours;
# 1.0 m leaves room for the ego's own lateral drift within a frame and for the
# ends of a polyline wandering as its far extension is refitted.
LANE_TRACK_GATE_M = 1.0

# Forward span two polylines must share before their offsets are compared.
#
# This has to move with _component_polyline. Tracing one branch instead of
# averaging across all of them makes polylines shorter -- a quarter of them now
# cover under 2 m of road -- and MAX_EXTEND_SPAN_MULTIPLE then keeps their
# extensions short too. At 5 m, 763 of 1067 adjacent-frame pairs on wrongway/4
# were rejected for insufficient overlap alone, association collapsed into
# 1-2 frame fragments, and 18 frames came back empty. At 2 m: 4 empty frames.
#
# What guards the association is not this but LANE_TRACK_GATE_M -- two markings
# must be within 1 m of each other laterally to be confused, and a lane is 3.5 m
# wide. This only has to stop two polylines that barely touch from voting.
MIN_LANE_OVERLAP_M = 2.0

# Frames a marking may go undetected and still be the same marking. Generous --
# 2 seconds at 10 Hz -- because the cost of the two errors is not symmetric.
# Too short and a marking's fragments never join into one track, so none of them
# reaches MIN_LANE_TRACK_LEN and the whole marking is dropped; too long and the
# worst case is a track that fails the lateral gate and simply does not join.
# Joining the wrong two markings needs them within LANE_TRACK_GATE_M of each
# other, which a 3.5 m lane rules out. Measured on wrongway/4: at a gap of 10
# the clip still has 10 empty frames, at 20 it has 4.
LANE_TRACK_MAX_GAP = 20

# Frames a track must appear in before it is believed at all. Set above the
# longest burst of consecutive false positives rather than below the shortest
# real marking: a real lane is in view for seconds.
MIN_LANE_TRACK_LEN = 5

# Spacing of the grid a filled polyline is built on, in metres.
LANE_FILL_STEP = 1.0


def _lane_offsets(polyline: np.ndarray, grid: np.ndarray):
    """Lateral offset of `polyline` at each x in `grid`, NaN outside its span.

    A marking is a function of distance ahead in the ego frame -- it has one
    lateral offset per forward distance -- which is what makes two of them
    comparable at all, and what a filled one is interpolated on.
    """
    order = np.argsort(polyline[:, 0])
    x, y = polyline[order, 0], polyline[order, 1]
    inside = (grid >= x[0]) & (grid <= x[-1])
    out = np.full(len(grid), np.nan)
    if inside.any():
        out[inside] = np.interp(grid[inside], x, y)
    return out


def _lane_distance(a: np.ndarray, b: np.ndarray, grid: np.ndarray) -> float:
    """Median lateral separation of two polylines over the road they share, or
    inf when they share too little of it to be compared."""
    offsets_a, offsets_b = _lane_offsets(a, grid), _lane_offsets(b, grid)
    shared = np.isfinite(offsets_a) & np.isfinite(offsets_b)
    if shared.sum() * LANE_FILL_STEP < MIN_LANE_OVERLAP_M:
        return np.inf
    return float(np.median(np.abs(offsets_a[shared] - offsets_b[shared])))


def _track_lanes(per_frame: list) -> list:
    """Greedy nearest-offset association down the clip.

    :param per_frame: one list of (N, 3) polylines per frame, in order
    :returns: tracks, each a dict of frame index -> polyline
    """
    grid = np.arange(0.0, EXTEND_RANGE + LANE_FILL_STEP, LANE_FILL_STEP)
    tracks, alive = [], []
    for index, polylines in enumerate(per_frame):
        taken, still_alive = set(), []
        # Nearest first, so the closest pairing claims its polyline before a
        # worse one can take it -- the same reason smooth_boxes.associate ranks
        # candidates rather than walking them in list order.
        pairs = sorted(
            ((_lane_distance(track["last"], polyline, grid), position, track)
             for track in alive for position, polyline in enumerate(polylines)),
            key=lambda pair: pair[0])
        claimed = set()
        for distance, position, track in pairs:
            if distance > LANE_TRACK_GATE_M or position in taken or id(track) in claimed:
                continue
            taken.add(position)
            claimed.add(id(track))
            track["frames"][index] = polylines[position]
            track["last"] = polylines[position]
            track["seen"] = index
        for track in alive:
            if index - track["seen"] <= LANE_TRACK_MAX_GAP:
                still_alive.append(track)
            else:
                tracks.append(track)
        for position, polyline in enumerate(polylines):
            if position not in taken:
                still_alive.append({"frames": {index: polyline},
                                    "last": polyline, "seen": index})
        alive = still_alive
    return tracks + alive


def _fill_track(track: dict, grid: np.ndarray) -> None:
    """Interpolates a track's polyline across the frames it is missing from.

    Between two observations the marking is the same piece of road seen from two
    places, so its offsets interpolate; outside them nothing is invented, which
    is why a track is never extended past its own first and last sighting.
    """
    observed = sorted(track["frames"])
    for before, after in zip(observed, observed[1:]):
        if after - before < 2:
            continue
        offsets_before = _lane_offsets(track["frames"][before], grid)
        offsets_after = _lane_offsets(track["frames"][after], grid)
        shared = np.isfinite(offsets_before) & np.isfinite(offsets_after)
        if not shared.any():
            continue
        for index in range(before + 1, after):
            weight = (index - before) / (after - before)
            offsets = (1.0 - weight) * offsets_before[shared] + \
                weight * offsets_after[shared]
            track["frames"][index] = np.stack(
                [grid[shared], offsets, np.zeros(shared.sum())], axis=1)


def stabilise_lanes(per_frame: list) -> list:
    """Drops markings the clip only glimpsed, and holds the rest across dropouts.

    :param per_frame: one list of (N, 3) ego-frame polylines per frame, in order
    :returns: the same, tracked; and a (dropped, filled) count for the report
    """
    grid = np.arange(0.0, EXTEND_RANGE + LANE_FILL_STEP, LANE_FILL_STEP)
    tracks = _track_lanes(per_frame)
    kept = [t for t in tracks if len(t["frames"]) >= MIN_LANE_TRACK_LEN]
    # By identity: a track holds numpy arrays, and `in` would compare them.
    keep_ids = {id(t) for t in kept}
    dropped = sum(len(t["frames"]) for t in tracks if id(t) not in keep_ids)
    filled = 0
    out = [[] for _ in per_frame]
    for track in kept:
        seen = set(track["frames"])
        _fill_track(track, grid)
        filled += len(track["frames"]) - len(seen)
        for index, polyline in track["frames"].items():
            out[index].append(polyline)
    return out, dropped, filled


def lane_polylines(mask: np.ndarray, points_cam: np.ndarray, intrinsics: np.ndarray,
                   calibration) -> list:
    """Lifts one frame's lane mask into 3D polylines.

    :param mask: boolean lane mask, (H, W), at the point map's resolution
    :param points_cam: UniDepth point map, (3, H, W), camera frame. Unused
        unless LANE_PLANE_FROM_DEPTH; may be None
    :param intrinsics: 3x3 pinhole intrinsics the point map is expressed in
    :param calibration: calibrate_ground.GroundCalibration for this clip
    :return: list of (N, 3) ego-frame polylines, ordered far-to-near
    """
    bridged = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, DASH_BRIDGE_KERNEL))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)

    # The calibrated road, or one fitted to the frame's own depth -- see
    # LANE_PLANE_FROM_DEPTH. Either way a single plane for the whole frame:
    # every marking in it lies on the same road surface.
    plane = None
    if LANE_PLANE_FROM_DEPTH and points_cam is not None:
        depth_valid = mask & (points_cam[2] > 1e-3)
        x_cam, y_cam, z_cam = points_cam[:, depth_valid]
        plane = _fit_ground_plane(calibration.camera_to_ego(
            np.stack([x_cam, y_cam, z_cam])))
    if plane is None:
        plane = (0.0, 0.0, 0.0)  # the calibrated road

    polylines = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < MIN_COMPONENT_PIXELS or \
                stats[label, cv2.CC_STAT_HEIGHT] < MIN_COMPONENT_ROWS:
            continue
        pixels = _component_polyline(labels == label)
        if len(pixels) < 2:
            continue
        origin, directions = _rays_to_ego(pixels, intrinsics, calibration)
        points = _intersect_ground(origin, directions, plane)
        if len(points) < 2:
            continue
        along, across = np.ptp(points[:, 0]), np.ptp(points[:, 1])
        if along < MIN_ALONG_ROAD_RATIO * across:
            continue
        # What the trace could not avoid: a line that wanders sideways further
        # than it travels forward is not a marking, whatever made it.
        ordered = points[np.argsort(points[:, 0])]
        wander = float(np.abs(np.diff(ordered[:, 1])).sum())
        if along > 1e-6 and wander / along > MAX_LANE_TORTUOSITY:
            continue
        polylines.append(points)

    # Rejoin before extending: two fragments of one marking give a far better
    # direction together than either does alone.
    return [_extend(lane, plane, intrinsics, calibration, mask.shape[:2])
            for lane in _merge_fragments(polylines)]
