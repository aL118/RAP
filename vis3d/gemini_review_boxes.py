#!/usr/bin/env python3
"""Review a clip's lifted 3D boxes with Gemini, and turn what it finds into edits.

    python gemini_review_boxes.py --clip ambulance
    python gemini_review_boxes.py --all --stride 10
    python gemini_review_boxes.py --clip closetruck \
        --note "frames 40-90: the box on the white pickup is a metre to its left"
    python gemini_review_boxes.py --dataset test --clip deer_family --find deer

Stage 3 fits every box independently from a monocular depth map, so a clip of
155 frames has 155 chances to put a box on a hedge, miss a parked car, or leave
a box sitting a metre off the vehicle it belongs to. Scrubbing all of them by
eye is the bottleneck this exists to remove: it renders each sampled frame with
the boxes drawn as numbered wireframes, asks Gemini which ones are wrong, and
writes the answers out as `manual_boxes_3d.json` edits that
`apply_manual_boxes_3d.py` merges in like hand-drawn ones.

WHAT GEMINI IS AND IS NOT ASKED

It is asked only about things visible in the image: does wireframe #3 sit on a
vehicle at all, does it cover that vehicle, does its front face point the way
the vehicle does, is there a vehicle with no wireframe on it.

It is NOT asked whether a box is the right size in metres, and the prompt never
shows it one. On these clips UniDepth's guessed intrinsics run ~3x long, so
every lifted box is metrically ~3x small -- a "car" is 2.5 x 1.0 m -- and the
error cancels on reprojection, which is exactly why the overlays look right.
Shown those numbers a reviewer would flag all 155 frames of every clip and be
correct about nothing that can be fixed here. See the lift's own size-prior
warning for the real thing, which is a stage-2 problem.

The same fact is what makes an image-space correction the right currency.
Because fx cancels, a box that covers its object in the overlay is as correct
as this pipeline can be, so Gemini returns the 2D box the wireframe *should*
have projected to and `fit_box_to_target` solves for the translation and scale
that puts it there. Nothing here ever asks a language model for a metre.

THREE OPERATIONS, ONE VOCABULARY

  fix     the box is on the right object but misplaced or mis-sized. Gemini
          returns `box_2d`, the object's true extent on screen; the solver
          translates the box in the camera's fronto-parallel plane and scales
          it about its own bottom face until its projected silhouette lands
          there. Bottom face, not centre: the wheel-contact row is the one
          well-measured thing about a lifted box (see calibrate_ground.py), so
          a resize must not lift the vehicle off the road.
  delete  the box is not on an object -- the ego bonnet, a hedge, a shadow.
  add     an object has no box. The 2D box's bottom edge is back-projected onto
          the frame's own ground height and the extent seeded from the median
          box of that class in this clip, so an added box inherits the clip's
          metric scale rather than a prior that disagrees with it.

PROPAGATION

A fault found once is a fault on every frame the object is in, and paying for
31 sampled frames only to fix 31 of 155 would waste most of what was found. So
each correction is re-expressed as a transform -- a pixel-space offset, a scale
factor, a yaw delta -- and pushed along that box's track, associated with
smooth_boxes.associate(). The offset is carried in PIXELS, converted back to
metres against each frame's own range: the fault being corrected is a mask that
sits off its object, which is constant on screen and therefore grows in metres
as the object recedes. A track reviewed at several frames is split at the
midpoints, so each frame takes the correction from the nearest frame that
actually saw it. --propagate frame turns all of this off.

YOUR OWN FINDINGS

--note text is put in front of the reviewer for the frames it covers, so a
fault you have already spotted is confirmed and turned into an edit rather than
re-discovered. A note may name its frames -- "frames 40-90: ..." or "frame 73:
..." -- and those frames are then reviewed whether or not --stride sampled
them. A note with no frames applies to the whole clip.

--find names a class outright, whether or not the detector has a label for it:
--find deer on a clip where stage 1 only ever found cars. It is a standing
instruction rather than a note because it changes what an `add` means -- the
40-pixel floor is lifted for the class asked for, and one finding is wanted per
animal rather than one per group. An added box for a class with no example
anywhere in the clip cannot copy a median, so its shape comes from PROPORTIONS
and its size from what the clip's own cars say a metre is here; see
seed_extent(), and the intrinsics paragraph above for why the two must come
from different places.

NOTHING IS APPLIED

This writes manual_boxes_3d.json and a report; it does not touch boxes_3d.json.
Read the report and the review_*.jpg images beside it, delete the edits you
disagree with, then run apply_manual_boxes_3d.py. Edits carry "by": "gemini",
and a re-run rewrites only those -- hand-made edits in the same file survive.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import box_schema  # noqa: E402
from smooth_boxes import associate  # noqa: E402

BASE = ROOT.parent
MANUAL = "manual_boxes_3d.json"

# --- what counts as an edit ----------------------------------------------------
# A "fix" whose target barely overlaps the box it claims to correct is far more
# likely a reviewer that mixed up two wireframes than a box that is 90% off, and
# acting on it moves a box that was right onto an object that already had one.
# Reported, not applied.
MIN_FIX_IOU = 0.05
# An "add" landing on top of an existing box is that box being re-reported as
# missing, which happens when a wireframe is small or mostly occluded.
MAX_ADD_IOU = 0.45
# How far the solver may take a box from where the lift put it. Both are per
# edit, not cumulative over a track.
MAX_SCALE, MIN_SCALE = 2.5, 0.4
# Solver iterations. The projection is near-linear over the distances involved,
# so this converges in three; six is for the near boxes where it is not.
FIT_ITERS = 6

# Radius for the position anchors written into manual_boxes_3d.json. Tighter
# than that file's own 3.0 m default: these anchors are generated from the very
# boxes_3d.json they will be applied to, so they should match to the metre --
# and a generous radius here would let one edit claim a neighbouring vehicle.
ANCHOR_RADIUS = 1.5

# Wireframe colours, BGR. Chosen to stay apart from each other and from tarmac,
# sky and foliage, because the reviewer's only handle on a box is which coloured
# wireframe carries which number.
PALETTE = [
    (60, 60, 255), (60, 220, 255), (60, 255, 60), (255, 200, 40),
    (255, 60, 200), (200, 120, 255), (255, 255, 60), (140, 200, 90),
    (40, 140, 255), (230, 130, 40),
]

# Front-facing edges of vehicle_corners_local's vertex order, and the rest.
# Drawn separately and thicker so the heading is legible: a wireframe cuboid
# without a marked front is symmetric, and half of what is being reviewed is
# whether the box is pointing the way the vehicle is.
CORNER_ORDER = np.array([
    [+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1],
    [+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1],
], dtype=np.float64) * 0.5
FRONT_EDGES = [(0, 1), (1, 5), (5, 4), (4, 0)]
OTHER_EDGES = [(2, 3), (3, 7), (7, 6), (6, 2),
               (0, 3), (1, 2), (4, 7), (5, 6)]

# Real-world length/width/height, used ONLY for their ratios to each other. An
# added box is seeded with the median box of its own class in the clip; a class
# the detector never found once -- which is the whole reason --find exists -- has
# no such median, so its shape comes from here and its size from the clip's own
# cars. Never from here directly: these are metres and a vis3d clip's boxes are
# not (see the module docstring), so a deer seeded at a literal 1.8 m would come
# out three times the size of the cars beside it.
PROPORTIONS = {
    "car": (4.5, 1.8, 1.5),      "truck": (7.0, 2.5, 3.2),
    "bus": (11.0, 2.6, 3.2),     "van": (5.2, 2.0, 2.2),
    "person": (0.6, 0.6, 1.7),   "pedestrian": (0.6, 0.6, 1.7),
    "bicycle": (1.7, 0.6, 1.1),  "motorcycle": (2.0, 0.8, 1.4),
    "deer": (1.8, 0.6, 1.4),     "dog": (1.0, 0.35, 0.7),
    "horse": (2.4, 0.8, 1.7),    "cow": (2.4, 0.9, 1.5),
}

HEADING_YAW = {
    "same_as_ego": 0.0,
    "oncoming": math.pi,
    "crossing_left_to_right": -math.pi / 2,
    "crossing_right_to_left": math.pi / 2,
}


# ------------------------------------------------------------------ geometry --
def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- renderer.rpy_to_rot in float64.

    Duplicated rather than imported: renderer.py lives under visualization/ and
    pulls in the whole rasterizer, and this module is run on a login node where
    that import is most of the startup.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def camera_matrices(camera: dict):
    """(ego->camera 4x4, intrinsics 3x3, (height, width)) out of a camera entry."""
    return (np.asarray(camera["ego_to_camera"], dtype=np.float64),
            np.asarray(camera["intrinsics"], dtype=np.float64),
            tuple(camera["image_hw"]))


def project_ego(points_ego, ego_to_camera, intrinsics):
    """(N,3) ego points -> (N,2) pixels and (N,) camera-frame depths."""
    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3).T
    cam = ego_to_camera[:3, :3] @ pts + ego_to_camera[:3, 3:4]
    depth = cam[2]
    uv = (intrinsics @ cam)[:2] / np.maximum(depth, 1e-9)
    return uv.T, depth


def box_corners_ego(box: np.ndarray) -> np.ndarray:
    """(9,) box row -> its 8 corners in the ego frame, in CORNER_ORDER order."""
    rot = rpy_to_rot(*box[box_schema.RPY])
    return (rot @ (CORNER_ORDER * box[box_schema.DIMS]).T).T + box[box_schema.CENTER]


def box_aabb_px(box: np.ndarray, ego_to_camera, intrinsics, image_hw):
    """Axis-aligned pixel extent of a box's projected corners, or None.

    None when the box is not usefully in front of the camera: any corner behind
    it makes the projection of that corner meaningless, and a box entirely
    outside the frame has no extent worth comparing a target against.
    """
    uv, depth = project_ego(box_corners_ego(box), ego_to_camera, intrinsics)
    if (depth < 0.5).any():
        return None
    height, width = image_hw
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    if x1 < 0 or y1 < 0 or x0 > width or y0 > height:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def aabb_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return float(inter / (area_a + area_b - inter + 1e-9))


def pixel_shift_to_ego(du: float, dv: float, point_ego, ego_to_camera, intrinsics):
    """Ego-frame translation that moves `point_ego` by (du, dv) pixels.

    Solved in the fronto-parallel plane at the point's own depth, so the move
    changes where the box appears without changing how far away it is: depth is
    the quantity the ground calibration fixed and the one a reviewer looking at
    a single image has no information about.
    """
    _, depth = project_ego([point_ego], ego_to_camera, intrinsics)
    z = float(depth[0])
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    delta_cam = np.array([du * z / fx, dv * z / fy, 0.0])
    return ego_to_camera[:3, :3].T @ delta_cam


def fit_box_to_target(box: np.ndarray, camera: dict, target_px):
    """Translate and scale `box` until its projected AABB matches `target_px`.

    Returns (fitted box, scale applied) or (None, reason) when the target asks
    for more than MIN_SCALE..MAX_SCALE, which means the reviewer was looking at
    a different object than the one it named.

    Scaling is about the bottom face -- the box grows upward and outward from
    the road, never off it -- and translation is fronto-parallel, so the box's
    range is left exactly as the lift and the ground calibration set it.

    The fit is allowed to move the box vertically in the image and the result is
    not re-seated on a ground plane afterwards. It does not need to be: a
    fronto-parallel move that puts the box's bottom edge on the vehicle's
    wheel-contact row puts its bottom face on the ray through that row at the
    box's own range, which is where the road is if the range is right. Re-seating
    would only overwrite the one vertical measurement the reviewer supplied with
    a median taken over the boxes it just called wrong.
    """
    ego_to_camera, intrinsics, image_hw = camera_matrices(camera)
    target = np.asarray(target_px, dtype=np.float64)
    target_wh = np.array([target[2] - target[0], target[3] - target[1]])
    target_centre = np.array([(target[0] + target[2]) / 2, (target[1] + target[3]) / 2])
    if (target_wh <= 1).any():
        return None, "degenerate target box"

    fitted = box.copy()
    total_scale = 1.0
    for _ in range(FIT_ITERS):
        aabb = box_aabb_px(fitted, ego_to_camera, intrinsics, image_hw)
        if aabb is None:
            return None, "box does not project into the frame"
        current_wh = np.array([aabb[2] - aabb[0], aabb[3] - aabb[1]])
        # The two ratios disagree whenever the target's aspect differs from the
        # cuboid's silhouette, which it always does a little: a cuboid's AABB
        # includes the corner nearest the camera and a vehicle's does not. The
        # mean splits the difference; matching either exactly would trade a
        # width error for a height one.
        step = float(np.mean(target_wh / np.maximum(current_wh, 1e-6)))
        step = float(np.clip(step, 0.5, 2.0))
        if MIN_SCALE <= total_scale * step <= MAX_SCALE:
            bottom = fitted[box_schema.Z] - fitted[box_schema.HEIGHT] / 2.0
            fitted[box_schema.DIMS] *= step
            fitted[box_schema.Z] = bottom + fitted[box_schema.HEIGHT] / 2.0
            total_scale *= step

        aabb = box_aabb_px(fitted, ego_to_camera, intrinsics, image_hw)
        if aabb is None:
            return None, "box left the frame while fitting"
        centre = np.array([(aabb[0] + aabb[2]) / 2, (aabb[1] + aabb[3]) / 2])
        du, dv = target_centre - centre
        fitted[box_schema.CENTER] += pixel_shift_to_ego(
            du, dv, fitted[box_schema.CENTER], ego_to_camera, intrinsics)
        if abs(du) < 1.0 and abs(dv) < 1.0 and abs(step - 1.0) < 0.01:
            break

    if not MIN_SCALE <= total_scale <= MAX_SCALE:
        return None, f"target needs a {total_scale:.2f}x resize"
    return fitted, total_scale


def ground_height(boxes: np.ndarray) -> float:
    """The frame's road height in the ego frame: the median box bottom.

    Read off the boxes rather than assumed to be 0, because the ground plane a
    clip was calibrated onto is only zero to within the calibration, and a box
    added on a plane a third of a metre off the others is visibly floating.
    """
    if len(boxes) == 0:
        return 0.0
    return float(np.median(boxes[:, box_schema.Z] - boxes[:, box_schema.HEIGHT] / 2.0))


def back_project_to_ground(u: float, v: float, camera: dict, ground_z: float):
    """The ego-frame point where the ray through pixel (u, v) meets z = ground_z."""
    ego_to_camera, intrinsics, _ = camera_matrices(camera)
    rot, trans = ego_to_camera[:3, :3], ego_to_camera[:3, 3]
    direction = rot.T @ (np.linalg.inv(intrinsics) @ np.array([u, v, 1.0]))
    origin = -rot.T @ trans
    if abs(direction[2]) < 1e-6:
        return None
    lam = (ground_z - origin[2]) / direction[2]
    if lam <= 0:
        return None                      # the ray meets the plane behind the camera
    return origin + lam * direction


def size_from_target(target_px, camera: dict, ground_z: float, name: str):
    """Extent for an added box, measured off the image rather than assumed.

    The 2D box's bottom edge back-projects to a point on the ground, which gives
    the object's range; its pixel height then gives the object's real height,
    `height = pixel_height * range / fy`. Length and width follow from the
    class's proportions *relative to that height*.

    Height is the quantity to measure and the other two the ones to infer,
    because height is the only extent that does not depend on which way the
    object is facing: a deer seen head-on and side-on is the same height and a
    very different width.

    This replaces seeding from `clip_scale`, which cannot be trusted on a clip
    whose detections are mostly false positives. On deer_family the boxes
    labelled "car" are the ego bonnet and a building; scaling a deer by their
    median made it 1.13x the length of a "car" -- a ratio that means nothing,
    since the denominator was a 2 m hood box. Everything here comes from this
    one object's own pixels and the same intrinsics the lift used, so it stays
    consistent with the clip whatever those intrinsics are.

    Returns None when the ray does not meet the ground in front of the camera.
    """
    _, intrinsics, _ = camera_matrices(camera)
    target = np.asarray(target_px, dtype=float)
    foot = back_project_to_ground((target[0] + target[2]) / 2.0, target[3],
                                  camera, ground_z)
    if foot is None:
        return None, None
    _, depth = project_ego([foot], *camera_matrices(camera)[:2])
    height = float(target[3] - target[1]) * float(depth[0]) / float(intrinsics[1, 1])
    length_p, width_p, height_p = PROPORTIONS.get(name, PROPORTIONS["car"])
    return foot, np.array([height * length_p / height_p,
                           height * width_p / height_p,
                           height])


def clip_scale(all_boxes: np.ndarray, all_names) -> float:
    """How many of this clip's units make one real metre, read off its own boxes.

    Every clip has its own factor because it is UniDepth's guessed focal length,
    and one guess per clip. Measured against whichever class in the frame has a
    known real-world length, cars first because there is nearly always one.
    """
    for reference in ("car", "truck", "bus", "van"):
        same = [b for b, n in zip(all_boxes, all_names) if n == reference]
        if same:
            median = float(np.median(np.stack(same)[:, box_schema.LENGTH]))
            return median / PROPORTIONS[reference][0]
    return 1.0


def seed_extent(name: str, all_boxes: np.ndarray, all_names) -> np.ndarray:
    """Starting length/width/height for an added box, in this clip's own units.

    Three sources, best first: the median box of the same class in this clip;
    that class's real-world proportions rescaled by what this clip's cars say a
    metre is; the median box of any class. Only the first needs no assumption,
    and it is the one that is missing exactly when --find is being used -- a
    class the detector never once found has no exemplar to copy.

    The fitter rescales whatever comes out of here to match the 2D box anyway,
    so what actually matters is the *shape*: a deer seeded with a car's
    proportions comes out long, wide and low however well its silhouette is
    matched, and needs a resize near the edge of MIN_SCALE to get there.
    """
    if len(all_boxes):
        same = [b for b, n in zip(all_boxes, all_names) if n == name]
        if same:
            return np.median(np.stack(same)[:, box_schema.DIMS], axis=0)
        if name in PROPORTIONS:
            return np.asarray(PROPORTIONS[name]) * clip_scale(all_boxes, all_names)
        return np.median(all_boxes[:, box_schema.DIMS], axis=0)
    return np.asarray(PROPORTIONS.get(name, PROPORTIONS["car"]))


# ------------------------------------------------------------------ rendering --
def draw_review_image(frame_bgr: np.ndarray, boxes: np.ndarray, names, camera: dict):
    """The frame with each box drawn as a numbered wireframe. Returns (image, drawn).

    `drawn` is the list of box indices that actually reached the canvas, and it
    is what the prompt enumerates: a reviewer asked about a box it cannot see
    invents an answer, and a box behind the camera or off the edge is not a
    review finding, it is a box that is not in this picture.

    Wireframes, not the filled cuboids of vis3d_overlay/: a filled box hides the
    vehicle underneath it, and every question being asked here is about how well
    the two line up.
    """
    ego_to_camera, intrinsics, image_hw = camera_matrices(camera)
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]
    drawn = []

    order = np.argsort([-np.hypot(b[box_schema.X], b[box_schema.Y]) for b in boxes])
    for index in order:                  # far to near, so near boxes draw on top
        box = boxes[index]
        uv, depth = project_ego(box_corners_ego(box), ego_to_camera, intrinsics)
        if (depth < 0.5).any():
            continue
        aabb = box_aabb_px(box, ego_to_camera, intrinsics, image_hw)
        if aabb is None:
            continue
        colour = PALETTE[int(index) % len(PALETTE)]
        points = np.clip(uv, -1e4, 1e4).astype(np.int32)
        for a, b in OTHER_EDGES:
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), colour, 2, cv2.LINE_AA)
        for a, b in FRONT_EDGES:
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), colour, 4, cv2.LINE_AA)
        _draw_tag(canvas, f"#{index}", aabb, colour, width, height)
        drawn.append(int(index))

    return canvas, sorted(drawn)


def _draw_tag(canvas, text, aabb, colour, width, height):
    """A filled chip carrying the box number, pinned inside the frame.

    Pinned rather than placed: the boxes worth reviewing are disproportionately
    the ones half out of frame, and a number drawn off-canvas is a wireframe the
    reviewer cannot refer to.
    """
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = int(np.clip(aabb[0], 2, width - tw - 10))
    y = int(np.clip(aabb[1] - 6, th + 8, height - 4))
    cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 8, y + 4), colour, -1)
    cv2.putText(canvas, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 2, cv2.LINE_AA)


# --------------------------------------------------------------------- prompt --
SYSTEM_INSTRUCTION = """\
You are reviewing 3D object boxes that an automatic pipeline fitted to dashcam
footage. Each box is drawn on the frame as a coloured wireframe cuboid carrying
its number, like #3. The four thick edges of a wireframe are the FRONT face of
the object the box claims to be on -- the end a vehicle drives towards.

Judge only what the picture shows:

  1. Is the wireframe on a real road user at all? A box on the ego vehicle's own
     bonnet, on a hedge, a shadow, a building, a parked bicycle rack, a patch of
     road, or on nothing, is wrong and should be deleted.
  2. Does the wireframe cover its object? The cuboid should enclose the vehicle:
     its bottom edges at the tyres, its top near the roof, its sides at the body.
     A box shifted off its vehicle, or much larger or smaller than it, is wrong
     and should be fixed.
  3. Does the thick front face point the way the object faces? A box whose front
     face is drawn across the side of a car has the heading wrong.
  4. Is there a clearly visible vehicle, pedestrian, cyclist or motorcyclist --
     or an object of any class the request below asks you to look for -- with NO
     wireframe on it? That one should be added. Only if it is unmistakable and
     not tiny: skip anything smaller than about 40 pixels across, anything more
     than about three quarters occluded, and anything beyond the traffic in the
     scene -- parked cars far up a side street are not worth adding.

Do NOT comment on:
  - how large a box is in metres, or how far away it is. You are not shown those
    numbers and they are on a scale of their own; a box that covers its object on
    screen is correct however implausible its size in metres would be.
  - boxes being semi-transparent, overlapping each other, or being drawn over
    each other. That is how the picture is rendered.
  - traffic lights, lane markings, or anything not a box.
  - the class label unless it is plainly wrong (a bus wireframed as a person).

Be strict about precision and conservative about volume. A frame where every box
sits on its vehicle is a frame with no findings, and saying so is the correct and
common answer. Do not invent a finding to have one. Report a box at most once.

When the request asks you to box a particular kind of object, that request is the
job and it comes first: box every one of them that is visible in this frame, each
as its own separate finding with its own box_2d, and use the word the request
uses as the name. Then carry on and report any other faults you see. An object of
that kind which already has a wireframe on it is not an add -- it is either
correct and silent, or a fix.

For a fix or an add, return box_2d: where the object's own 2D bounding box is in
this image, as [ymin, xmin, ymax, xmax] normalised to 0-1000. Draw it around the
visible extent of the object itself, not around the wireframe. Get it tight --
it is used directly to move the 3D box, so a loose box_2d makes a loose 3D box.

confidence is 0-1: how sure you are that this is a real mistake worth an edit.
reason is one short clause naming the object and the fault, e.g. "box sits a car
width left of the white pickup"."""

BOX3D_SCHEMA = {
    "type": "object",
    "description": "the 3D box, ego frame: x forward, y left, z up, metres; "
                   "yaw radians counter-clockwise from the ego heading",
    "properties": {
        "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
        "length": {"type": "number"}, "width": {"type": "number"},
        "height": {"type": "number"}, "yaw": {"type": "number"},
    },
    "required": ["x", "y", "z", "length", "width", "height", "yaw"],
}

BOX2D_SCHEMA = {
    "type": "array", "items": {"type": "integer"},
    "description": "[ymin, xmin, ymax, xmax], normalised to 0-1000",
}

HEADING_SCHEMA = {"type": "string",
                  "enum": ["unchanged", "same_as_ego", "oncoming",
                           "crossing_left_to_right", "crossing_right_to_left"]}


def response_schema(boxes_from: str) -> dict:
    """The output schema, with each operation's own fields actually required.

    Split into one array per operation rather than a single `findings` list with
    an `op` field, because a schema can only mark a field required for every
    item of an array or for none of them -- and what an add must carry is not
    what a delete must. Under the flat version `box3d` was merely *described*
    and asked for in the prompt, so the model was free to leave it out, and on
    deer_family it left it out of all 17 deer it found: seventeen correct
    sightings at 0.80-0.95 confidence, every one dropped for having no box.
    A field the pipeline cannot proceed without belongs in `required`, where the
    API enforces it, not in the prose where it is a request.

    Which geometry field is required depends on the mode, since only one of them
    is ever used: box3d under --boxes_from model, box_2d under fit.
    """
    geometry = ({"box3d": BOX3D_SCHEMA} if boxes_from == "model"
                else {"box_2d": BOX2D_SCHEMA})
    geometry_key = next(iter(geometry))
    common = {"confidence": {"type": "number"}, "reason": {"type": "string"}}

    return {
        "type": "object",
        "properties": {
            "deletes": {
                "type": "array",
                "description": "boxes that are not on a real object and should go",
                "items": {"type": "object",
                          "properties": {"box": {"type": "integer"}, **common},
                          "required": ["box", "confidence", "reason"]},
            },
            "adds": {
                "type": "array",
                "description": "objects with no box on them at all",
                "items": {"type": "object",
                          "properties": {"name": {"type": "string"},
                                         "heading": HEADING_SCHEMA,
                                         **geometry, **common},
                          "required": ["name", geometry_key, "confidence", "reason"]},
            },
            "fixes": {
                "type": "array",
                "description": "boxes on the right object but misplaced or mis-sized",
                "items": {"type": "object",
                          "properties": {"box": {"type": "integer"},
                                         "name": {"type": "string"},
                                         "heading": HEADING_SCHEMA,
                                         **geometry, **common},
                          "required": ["box", geometry_key, "confidence", "reason"]},
            },
            "summary": {"type": "string"},
        },
        "required": ["deletes", "adds", "fixes", "summary"],
    }


def flatten(answer: dict) -> list:
    """The per-operation arrays -> the flat finding list the rest of this reads.

    Also accepts the older single `findings` array, so a findings.json saved
    before the split can still be re-derived with --from_findings.
    """
    if "findings" in answer:
        return list(answer.get("findings") or [])
    flat = []
    for key, op in (("deletes", "delete"), ("adds", "add"), ("fixes", "fix")):
        for item in answer.get(key) or []:
            entry = dict(item)
            entry["op"] = op
            entry.setdefault("box", -1)
            flat.append(entry)
    return flat


def build_prompt(drawn, names, image_hw, notes, find=(), boxes=None,
                 show_numbers=False) -> str:
    height, width = image_hw
    lines = [f"Dashcam frame, {width}x{height}. Wireframes drawn on it:"]
    if drawn:
        for i in drawn:
            label = f"  #{i} labelled '{names[i] if i < len(names) else '?'}'"
            if show_numbers and boxes is not None and i < len(boxes):
                b = boxes[i]
                label += (f"   x={b[box_schema.X]:.1f} y={b[box_schema.Y]:.1f} "
                          f"z={b[box_schema.Z]:.1f}  length={b[box_schema.LENGTH]:.1f} "
                          f"width={b[box_schema.WIDTH]:.1f} "
                          f"height={b[box_schema.HEIGHT]:.1f}  "
                          f"yaw={b[box_schema.YAW]:.2f}")
            lines.append(label)
    else:
        lines.append("  (none -- every box this frame is behind or outside the camera)")
    if show_numbers and drawn:
        # Only shown when the reviewer has to write numbers of its own. These are
        # the calibration: this pipeline's boxes are on a scale set by a focal
        # length guessed per clip, so the same real deer is ~1.4 tall in one
        # clip's units and a third of that in another's. A box3d written in
        # real-world metres would land right in the first and be three times too
        # big in the second, and there is nothing in the image that says which
        # clip this is. There is in these rows.
        lines += ["",
                  "Those are this pipeline's own numbers for the boxes you can see, "
                  "in the same frame and units your box3d must use. Read them as "
                  "the scale of this clip -- what a car-sized thing at that "
                  "distance measures here -- and put your own boxes on it. Do not "
                  "report them as right or wrong; they are reference, not work."]
    lines.append("")
    if find:
        wanted = ", ".join(find)
        lines += [f"REQUEST: box every {wanted} visible in this frame that does not "
                  f"already have a wireframe on it. Return one 'add' finding per "
                  f"{'animal or object' if len(find) > 1 else find[0]}, with "
                  f"name set to that word. Do this as well as, not instead of, "
                  f"reporting faults in the wireframes already drawn.",
                  "The 40-pixel floor does not apply to this request: box them "
                  "however small and distant they are, as long as you can tell "
                  "what they are and can put a tight box_2d around each one "
                  "separately. If several stand close together, that is several "
                  "findings, not one box around the group.",
                  "Set `heading` on every one of them, from the direction it is "
                  "facing in this frame -- crossing_left_to_right, "
                  "crossing_right_to_left, oncoming, or same_as_ego. An animal "
                  "crossing a road is not facing along it, and heading is the one "
                  "thing about an added box that cannot be recovered from its "
                  "box_2d, so omitting it leaves the box square to the road.", ""]
    if notes:
        # Ahead of the task, not appended to it: these are the faults the person
        # running this already found by eye, and they are the reason the frame is
        # being looked at at all.
        lines.append("The person who owns this footage reports the following about "
                     "this frame. Treat it as ground truth, confirm it against the "
                     "image, and return the edits that carry it out:")
        lines += [f"  - {n}" for n in notes]
        lines.append("")
    lines.append("Review these wireframes and report only real mistakes. "
                 "Return an empty findings list if they are all correct.")
    return "\n".join(lines)


# ------------------------------------------------------------------ the model --
# A 429 carries the wait the server actually wants ("Please retry in 42.7s"),
# which on the free tier -- 5 requests a minute -- is an order of magnitude more
# than any backoff worth writing by hand. Honouring it turns a run that dies at
# frame four into one that finishes slowly.
RETRY_HINT = re.compile(r"retry in ([\d.]+)\s*s", re.I)
QUOTA_ERROR = re.compile(r"429|quota|rate.?limit|too_many_requests", re.I)

# Consecutive frames that exhaust their retries on a quota error before the run
# gives up. A per-minute limit lets a frame through eventually and resets this;
# an exhausted daily allowance never does, and without a stop the run spends
# retries x frames x the API's own 55-second retry hint discovering that -- half
# an hour of sleeping to produce an empty report.
QUOTA_STRIKES = 3

# A truncated or malformed answer is not a rate limit, and must not be retried
# like one. Frame 000088 of deer_family burned five requests re-asking a
# question whose answer had been cut off mid-string -- a quarter of a free
# tier's daily allowance on one frame. Two attempts: sampling varies, so a
# second is worth having, a fifth is not.
PARSE_RETRIES = 2


def retry_delay(exc, attempt: int) -> float:
    hint = RETRY_HINT.search(str(exc))
    if hint:
        return min(float(hint.group(1)) + 1.0, 120.0)
    return min(2.0 * 2 ** attempt, 60.0)


class Throttle:
    """A request-per-minute ceiling shared by every worker thread.

    Retrying into a limit still spends the request that the limit rejected, so a
    run against a quota that is known in advance is far faster held below it than
    bounced off it. --rpm 0 leaves it out of the way entirely.
    """

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            due = max(now, self.next_at)
            self.next_at = due + self.interval
        if due > now:
            time.sleep(due - now)


class Reviewer:
    """One Gemini client, shared across the worker threads."""

    def __init__(self, model: str, retries: int = 3, rpm: float = 0.0,
                 system: str = SYSTEM_INSTRUCTION, max_output_tokens: int = 4096,
                 thinking: str = "low", boxes_from: str = "fit"):
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit(
                "error: GEMINI_API_KEY is not set.\n"
                "       export it, or run through scripts/vis3d/run_gemini_review.sh")
        from google import genai            # imported here so --help needs no SDK
        self.client = genai.Client()
        self.model = model
        self.system = system
        self.retries = retries
        self.throttle = Throttle(rpm)
        self.schema = response_schema(boxes_from)
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking
        self.strikes = 0
        self.halted = ""
        self.lock = threading.Lock()

    def review(self, image_bgr: np.ndarray, prompt: str) -> dict:
        with self.lock:
            if self.halted:
                raise RuntimeError(self.halted)
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("cv2.imencode failed on the review image")
        payload = base64.b64encode(buf.tobytes()).decode("utf-8")

        last = None
        for attempt in range(self.retries):
            self.throttle.wait()
            try:
                result = self.client.interactions.create(
                    model=self.model,
                    system_instruction=self.system,
                    # schema_, not schema: the SDK's TypedDict spells it with the
                    # trailing underscore and serialises it back to "schema", and
                    # a plain "schema" key is dropped on the floor -- the request
                    # then fails with "responseFormat must be set", which reads
                    # like the whole field was missing rather than one key of it.
                    response_format={"type": "text", "mime_type": "application/json",
                                     "schema_": self.schema},
                    # Explicit, because the default budget is shared with the
                    # model's own thinking: a frame with several findings had its
                    # JSON cut off mid-string, which arrives here as a parse
                    # error rather than as anything mentioning a limit.
                    generation_config={
                        "max_output_tokens": self.max_output_tokens,
                        "thinking_level": self.thinking,
                    },
                    input=[{"type": "text", "text": prompt},
                           {"type": "image", "data": payload, "mime_type": "image/jpeg"}],
                )
                parsed = _parse_json(result.output_text)
                with self.lock:
                    self.strikes = 0
                return parsed
            except Exception as exc:      # noqa: BLE001 -- rate limits, 5xx, bad JSON
                last = exc
                budget = (PARSE_RETRIES if isinstance(exc, json.JSONDecodeError)
                          else self.retries)
                if attempt >= budget - 1:
                    break
                time.sleep(retry_delay(exc, attempt))

        if QUOTA_ERROR.search(str(last)):
            with self.lock:
                self.strikes += 1
                if self.strikes >= QUOTA_STRIKES and not self.halted:
                    self.halted = (
                        f"stopping: {QUOTA_STRIKES} frames in a row exhausted their "
                        f"retries on a quota error, so this is an allowance that has "
                        f"run out rather than a pace that is too fast. Lower --rpm if "
                        f"it is a per-minute limit, or check the key's quota. Last "
                        f"error: {last}")
        raise RuntimeError(f"gemini failed after {self.retries} tries: {last}")


def load_system(path) -> str:
    """The review rules, read from a file instead of the constant above.

    Kept overridable because the rules are the part of this that gets tuned --
    what counts as a fault, what to leave alone, how strict to be -- and tuning
    them should not mean editing Python. run_gemini.sh carries its copy as a
    heredoc so a run and the instructions it ran under sit in one file.
    """
    text = Path(path).read_text().strip()
    if not text:
        raise SystemExit(f"error: {path} is empty; the review rules cannot be blank")
    return text


def _parse_json(text: str) -> dict:
    """Tolerant of a fenced block, which structured output should not produce
    but does when a request is served without the schema attached."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def box_from_model(box3d):
    """A model-supplied box3d -> a (9,) row, or None with a reason.

    roll and pitch are filled with zero rather than required. The lift leaves
    them at zero on every box it writes -- only yaw is ever estimated -- so
    asking a reviewer for two angles that are always zero is two more chances
    for it to put something else there.

    The checks are for physical nonsense only, not for plausibility: a box
    behind the camera or with a negative extent cannot be rendered or exported
    at all, while a box that is merely the wrong size is exactly the kind of
    thing the reviewer is being trusted to judge in this mode.
    """
    if not isinstance(box3d, dict):
        return None, "no box3d"
    row = np.zeros(box_schema.BOX_DIM)
    try:
        for key in ("x", "y", "z", "length", "width", "height", "yaw"):
            row[box_schema.KEYS.index(key)] = float(box3d[key])
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"malformed box3d ({exc})"
    if not np.isfinite(row).all():
        return None, "box3d holds a non-finite number"
    if (row[box_schema.DIMS] <= 0).any():
        return None, "box3d has a non-positive extent"
    if row[box_schema.X] <= 0:
        return None, f"box3d sits behind the camera (x={row[box_schema.X]:.1f})"
    return row, ""


def denormalise(box_2d, image_hw):
    """Gemini's [ymin, xmin, ymax, xmax] in 0-1000 -> pixel [x0, y0, x1, y1].

    That order and that scale are the convention Gemini's own box outputs are
    trained on, so it is what the prompt asks for even though everything else
    here is in pixels. Values are clamped rather than rejected: a box clipped by
    the frame edge comes back slightly out of range, and that is the box being
    reported, not a malformed answer.
    """
    if not box_2d or len(box_2d) != 4:
        return None
    height, width = image_hw
    ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in box_2d)
    x0, x1 = sorted((xmin * width, xmax * width))
    y0, y1 = sorted((ymin * height, ymax * height))
    return np.array([max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)])


# ------------------------------------------------------------------ your notes --
# "frames 40-90: ...", "frame 73: ...", "f000073 ...", "40-90 ..." -- all of which
# people actually write. A note that names no frame applies to the whole clip.
NOTE_SCOPE = re.compile(
    r"^\s*(?:frames?\s*)?((?:f?\d+(?:\s*-\s*f?\d+)?)(?:\s*,\s*f?\d+(?:\s*-\s*f?\d+)?)*)\s*[:\-]\s*(.+)$",
    re.S | re.I)


def parse_frame_spec(spec: str) -> set:
    """"70-110,150" -> {70, ..., 110, 150}. Frame *numbers*, not list indices."""
    numbers = set()
    for part in str(spec).split(","):
        part = part.strip().lstrip("fF")
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(p.strip().lstrip("fF")) for p in part.split("-", 1))
            numbers.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            numbers.add(int(part))
    return numbers


def parse_note(text: str):
    """One note -> (frame numbers it covers or None, the note itself)."""
    match = NOTE_SCOPE.match(text.strip())
    if not match:
        return None, text.strip()
    try:
        return parse_frame_spec(match.group(1)), match.group(2).strip()
    except ValueError:
        return None, text.strip()


def load_notes(args, clip: str):
    """Notes that apply to `clip`, from --note and --notes_file.

    A notes file may address a clip by name with a `[clip]` heading, so one file
    can carry a review pass over the whole dataset; lines before any heading
    apply to every clip, which is where a systematic complaint goes.
    """
    raw = list(args.note or [])
    if args.notes_file:
        current = None
        for line in Path(args.notes_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            heading = re.match(r"^\[(.+?)\]$", line)
            if heading:
                current = heading.group(1).strip()
                continue
            if current in (None, clip):
                raw.append(line)
    return [parse_note(n) for n in raw]


# ------------------------------------------------------------------ propagation --
def track_index(frames, data, max_dist, max_gap, max_px):
    """(frame index, slot) -> track id, and track id -> [(frame index, slot), ...].

    Straight out of smooth_boxes.associate, which is the association this
    pipeline already trusts to decide which boxes are the same object; running a
    second, different one here would let a correction propagate along a track
    that the smoother does not believe in.
    """
    tracks = associate(frames, data, max_dist, max_gap, max_px=max_px)
    of_box, members = {}, {}
    for ti, track in enumerate(tracks):
        members[ti] = list(zip(track["idx"], track["slot"]))
        for fi, slot in members[ti]:
            of_box[(fi, slot)] = ti
    return of_box, members


def scale_about_bottom(box: np.ndarray, scale: float) -> np.ndarray:
    out = box.copy()
    bottom = out[box_schema.Z] - out[box_schema.HEIGHT] / 2.0
    out[box_schema.DIMS] *= scale
    out[box_schema.Z] = bottom + out[box_schema.HEIGHT] / 2.0
    return out


def apply_transform(box: np.ndarray, transform: dict, camera: dict) -> np.ndarray:
    """Re-play a fix on another frame's box: scale, then yaw, then pixel offset.

    The offset is carried in pixels and converted here against *this* frame's
    range, so a box that sat half a car-width off its vehicle keeps sitting half
    a car-width off it as the vehicle recedes -- which is what a mask that is
    consistently misaligned actually does. Carrying the metres instead would
    over-correct everything nearer than the frame the fault was found on and
    under-correct everything further.
    """
    ego_to_camera, intrinsics, _ = camera_matrices(camera)
    out = scale_about_bottom(box, transform.get("scale", 1.0))
    if transform.get("yaw") is not None:
        out[box_schema.YAW] = transform["yaw"]
    du, dv = transform.get("shift_px", (0.0, 0.0))
    if du or dv:
        out[box_schema.CENTER] += pixel_shift_to_ego(
            du, dv, out[box_schema.CENTER], ego_to_camera, intrinsics)
    return out


def _projected_aabb(box, camera):
    """Where a box lands on screen, as a drawable [x0, y0, x1, y1] or None.

    In --boxes_from model this is the only check on the reviewer's arithmetic
    that costs nothing: the annotated frame draws it, so a box whose numbers put
    it in the sky or a hundred metres past its animal is visible at a glance
    rather than only after a re-render.
    """
    aabb = box_aabb_px(box, *camera_matrices(camera))
    return None if aabb is None else [round(float(v), 1) for v in aabb]


def aabb_centre(box, camera):
    ego_to_camera, intrinsics, image_hw = camera_matrices(camera)
    aabb = box_aabb_px(box, ego_to_camera, intrinsics, image_hw)
    if aabb is None:
        return None
    return np.array([(aabb[0] + aabb[2]) / 2.0, (aabb[1] + aabb[3]) / 2.0])


# ------------------------------------------------------------------ edit build --
def make_edit(op: str, anchor=None, box=None, name=None, score=None,
              why: str = "", frame: str = "", origin: str = ""):
    """One manual_boxes_3d.json edit, tagged so a re-run can find its own work.

    `why`/`by`/`from` are ignored by apply_manual_boxes_3d.py, which reads only
    op/at/radius/box/name/score. They are here because a file of forty anonymous
    coordinate triples is not reviewable, and reviewing this file before applying
    it is the entire safeguard.
    """
    edit = {"op": op, "by": "gemini"}
    if anchor is not None:
        edit["at"] = [round(float(v), 4) for v in anchor]
        edit["radius"] = ANCHOR_RADIUS
    if box is not None:
        edit["box"] = box_schema.to_dict(box)
    if name is not None:
        edit["name"] = name
    if score is not None:
        edit["score"] = float(score)
    if why:
        edit["why"] = why
    if origin:
        edit["from"] = origin
    return edit


def link_adds(adds, names_by_key, max_gap=10, per_frame_lengths=0.4,
              base_lengths=0.75):
    """Chain added boxes across reviewed frames into tracks. Returns list of chains.

    An `add` is anchored to one reviewed frame, so on a --stride 8 pass a deer
    the detector never saw would come back as isolated boxes eight frames apart
    and blink. Two adds of the same class are taken to be the same object when
    they are close enough, and the frames between them are then interpolated --
    the same keyframe rule apply_manual_boxes.py uses for hand-drawn tracks.

    "Close enough" has to know how many frames apart the two adds are. A single
    distance gate is wrong in both directions at once: wide enough for a car
    closing at 15 m/s across an eight-frame gap, it also swallows the deer
    standing beside the one it should have matched. So the gate is the object's
    own length times `base_lengths + per_frame_lengths * gap` -- it opens as the
    gap grows and stays shut across a small one.

    `max_gap` is the hard stop on that opening, and it matters more than the
    shape of the gate. A distance gate that keeps widening will happily chain
    two adds forty frames apart and then interpolate four seconds of animal
    motion out of two endpoints -- 6 observed deer became 83 boxes before this
    existed. Past `max_gap` the keyframes stay separate: a box on each frame
    that was actually looked at, and nothing invented in between. Which means a
    --stride coarser than this fills no gaps at all, and should not: nothing was
    seen there.

    Assignment within a frame is nearest-first over every (add, chain) pair
    rather than first-come, for the reason smooth_boxes.associate resolves its
    pairs the same way: with three deer in a frame, first-fit gives the first
    chain whichever deer it happens to see first.
    """
    by_frame = {}
    for key in sorted(adds):
        by_frame.setdefault(key[0], []).append(key)

    chains = []
    for fi in sorted(by_frame):
        pairs = []
        for key in by_frame[fi]:
            box, name = adds[key], names_by_key[key]
            for ci, chain in enumerate(chains):
                last_fi, last_box, last_name = chain[-1]
                if last_name != name or last_fi >= fi or fi - last_fi > max_gap:
                    continue
                length = max(float(box[box_schema.LENGTH]),
                             float(last_box[box_schema.LENGTH]))
                gate = length * (base_lengths + per_frame_lengths * (fi - last_fi))
                distance = float(np.linalg.norm(
                    last_box[box_schema.CENTER] - box[box_schema.CENTER]))
                if distance <= gate:
                    pairs.append((distance / gate, key, ci))
        pairs.sort(key=lambda pair: pair[0])

        used_keys, used_chains = set(), set()
        for _, key, ci in pairs:
            if key in used_keys or ci in used_chains:
                continue
            used_keys.add(key)
            used_chains.add(ci)
            chains[ci].append((fi, adds[key], names_by_key[key]))
        for key in by_frame[fi]:
            if key not in used_keys:
                chains.append([(fi, adds[key], names_by_key[key])])
    return chains


def interpolate_chain(chain, frames_wanted):
    """Boxes for the frames between a chain's keyframes. Yields (frame index, box)."""
    for (fi_a, box_a, _), (fi_b, box_b, _) in zip(chain, chain[1:]):
        for fi in frames_wanted:
            if not fi_a < fi < fi_b:
                continue
            t = (fi - fi_a) / float(fi_b - fi_a)
            row = box_a * (1.0 - t) + box_b * t
            # Yaw on the doubled angle, as smooth_boxes does: the lifter's yaw is
            # only defined up to +-pi, so a straight average can swing a box
            # through a half turn between two keyframes that agree.
            z = ((1.0 - t) * np.exp(2j * box_a[box_schema.YAW])
                 + t * np.exp(2j * box_b[box_schema.YAW]))
            row[box_schema.YAW] = np.angle(z) / 2.0
            yield fi, row


# ------------------------------------------------------------------- the clip --
def select_frames(frames, args, notes):
    """Which frames get looked at: the stride, plus everything a note names.

    A note's frames are added whatever the stride is. Someone who writes "frame
    73: the box on the van is too far right" has looked at frame 73, and a
    sampler that skips it turns a report into nothing.
    """
    numbers = [int(Path(f).stem) for f in frames]
    wanted = set()
    if args.frames:
        asked = parse_frame_spec(args.frames)
        wanted |= {i for i, n in enumerate(numbers) if n in asked}
    else:
        wanted |= set(range(0, len(frames), max(1, args.stride)))
    for scope, _ in notes:
        if scope:
            wanted |= {i for i, n in enumerate(numbers) if n in scope}
    chosen = sorted(wanted)
    if args.limit:
        chosen = chosen[:args.limit]
    return chosen


def notes_for_frame(notes, frame_number):
    return [text for scope, text in notes
            if scope is None or frame_number in scope]


def annotate_findings(image, decided, image_hw):
    """Draws what the reviewer asked for over the wireframes it was shown.

    Red is a box to be deleted, white is a box to be added or moved, grey is a
    finding that was dropped -- so a glance at the saved image says what will
    change, in which direction, and what was rejected on your behalf, which is
    the thing a report of coordinates cannot show.
    """
    canvas = image.copy()
    for entry in decided:
        target = entry.get("target_px")
        if target is None:
            continue
        if not entry["applied"]:
            colour = (150, 150, 150)
        elif entry["op"] == "delete":
            colour = (0, 0, 255)
        else:
            colour = (255, 255, 255)
        p0 = (int(target[0]), int(target[1]))
        p1 = (int(target[2]), int(target[3]))
        cv2.rectangle(canvas, p0, p1, colour, 2, cv2.LINE_AA)
        label = f"{entry['op']} #{entry['box']}" if entry["box"] >= 0 else f"{entry['op']}"
        cv2.putText(canvas, label, (p0[0] + 3, max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (p0[0] + 3, max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return canvas


def cached_findings(review_dir: Path, frames, in_boxes: str = "boxes_3d.json"):
    """A previous run's raw answers, keyed by frame index. {} if there are none.

    findings.json holds what the model said, not what was done with it, so a
    second pass over it can apply different gates, a different --max_add_gap or
    a different --propagate and get a different -- and free -- answer. On a key
    limited to 20 requests a day, re-deriving beats re-asking.

    A finding names a box by the NUMBER drawn on it, which is its index in the
    file that was reviewed. Replaying answers about one file against a different
    one therefore retargets every finding at whatever now sits at that index --
    silently, and catastrophically: replaying a lift's answers onto the file
    those same answers already corrected deletes a second box for every delete.
    So the source is recorded when the cache is written, and a mismatch stops
    the run rather than being repaired by guesswork.
    """
    path = review_dir / "findings.json"
    if not path.exists():
        return {}
    saved = json.loads(path.read_text())
    reviewed = saved.get("_reviewed", "boxes_3d.json")
    if reviewed != in_boxes:
        raise SystemExit(
            f"error: {path} holds answers about {reviewed}, but this run reviews\n"
            f"       {in_boxes}. A finding names a box by its index in the file it\n"
            f"       was shown, so replaying these would edit whatever now sits at\n"
            f"       that index. Re-run with --in_boxes {reviewed}, or drop\n"
            f"       --from_findings to ask again about {in_boxes}.")
    index = {frame: fi for fi, frame in enumerate(frames)}
    return {index[frame]: entry for frame, entry in saved.items()
            if frame in index and "findings" in entry}


def review_one_frame(job):
    """Render, ask, decide -- the whole of one frame, run on a worker thread."""
    (fi, frame, entry, frames_dir, reviewer, notes, args, cached) = job
    image_path = Path(frames_dir) / frame
    image = cv2.imread(str(image_path))
    if image is None:
        return fi, {"error": f"cannot read {image_path}"}

    boxes = box_schema.to_array(entry.get("boxes", []))
    names = list(entry.get("names", []))
    camera = entry.get("camera")
    if camera is None:
        return fi, {"error": "frame has no camera entry"}
    _, _, image_hw = camera_matrices(camera)

    review_image, drawn = draw_review_image(image, boxes, names, camera)
    if cached is not None:
        answer = cached
    else:
        prompt = build_prompt(drawn, names, image_hw, notes, args.find, boxes,
                              show_numbers=args.boxes_from == "model")
        try:
            answer = reviewer.review(review_image, prompt)
        except Exception as exc:          # noqa: BLE001
            return fi, {"error": str(exc)}

    decided = decide(flatten(answer), boxes, names, camera, drawn, args)
    return fi, {"summary": answer.get("summary", ""), "decided": decided,
                "image": review_image, "boxes": boxes, "camera": camera}


def decide(findings, boxes, names, camera, drawn, args):
    """Turn raw findings into accepted corrections, with a reason for each drop.

    Everything the reviewer said is kept, applied or not: a rejection is the
    interesting half of a review pass, since a run that quietly drops half its
    findings and a run that found half as many read identically in the output.
    """
    ego_to_camera, intrinsics, image_hw = camera_matrices(camera)
    current = {i: box_aabb_px(boxes[i], ego_to_camera, intrinsics, image_hw)
               for i in drawn}
    ground_z = ground_height(boxes)
    decided, seen = [], set()
    # Accepted adds are folded back into `current` as they are taken, so a second
    # add on the same object is caught by the same overlap test that catches an
    # add on an existing box. --find asks for one finding per object and a frame
    # with three deer in it is a frame where two of them look alike.
    added = []

    for finding in findings:
        op = str(finding.get("op", "")).lower()
        index = int(finding.get("box", -1))
        confidence = float(finding.get("confidence", 0.0))
        entry = {"op": op, "box": index, "confidence": confidence,
                 "reason": str(finding.get("reason", "")).strip(),
                 # The model's own answer, verbatim. target_px below is derived
                 # from it, but only box_2d/name/heading can re-derive a finding
                 # from scratch -- which is what --from_findings does, and what
                 # makes re-tuning propagation free.
                 "box_2d": finding.get("box_2d"),
                 "name": finding.get("name"),
                 "heading": finding.get("heading"),
                 "applied": False, "why_not": "", "target_px": None}
        decided.append(entry)

        target = denormalise(finding.get("box_2d"), image_hw)
        if target is None and finding.get("target_px"):
            # A findings.json written before box_2d was kept. target_px is that
            # same box already in pixels, so a cached run stays re-derivable
            # rather than being lost to the format it was saved in.
            target = np.asarray(finding["target_px"], dtype=float)
        if target is not None:
            entry["target_px"] = [round(float(v), 1) for v in target]

        if args.ops and op not in args.ops:
            entry["why_not"] = f"op {op!r} not in --ops"
            continue
        if confidence < args.min_confidence:
            entry["why_not"] = f"confidence {confidence:.2f} < {args.min_confidence:.2f}"
            continue
        if op in ("fix", "delete"):
            if index not in current or current[index] is None:
                entry["why_not"] = "no such wireframe in this frame"
                continue
            if (op, index) in seen:
                entry["why_not"] = "box already had a finding this frame"
                continue
            seen.add((op, index))

        if op == "delete":
            entry["applied"] = True
            entry["anchor"] = boxes[index][box_schema.CENTER].tolist()
            # The box being removed IS the target, so a delete shows up in the
            # annotated frame like every other finding. Without this a flagged
            # box looks exactly like an accepted one, and the reviewer's most
            # confident calls -- boxes on the ego bonnet -- are the invisible ones.
            entry["target_px"] = [round(float(v), 1) for v in current[index]]

        elif op == "fix":
            if args.boxes_from == "model":
                fitted, why = box_from_model(finding.get("box3d"))
                if fitted is None:
                    entry["why_not"] = why
                    continue
                entry.update({
                    "applied": True,
                    "anchor": boxes[index][box_schema.CENTER].tolist(),
                    "box": index, "fitted": fitted, "source": "model",
                    "name": str(finding.get("name") or
                               (names[index] if index < len(names) else "car")),
                    # No transform: a box the reviewer placed outright says
                    # nothing about how to move the same object on another
                    # frame, so it stays on the frame it was given for. See
                    # --propagate under boxes_from model.
                    "transform": None,
                })
                entry["target_px"] = _projected_aabb(fitted, camera)
                continue
            if target is None:
                entry["why_not"] = "fix with no box_2d"
                continue
            iou = aabb_iou(current[index], target)
            if iou < MIN_FIX_IOU:
                # Far more often a mixed-up wireframe number than a box that is
                # 95% off its object; either way, moving a box onto something a
                # long way from where it was is not a correction anyone asked for.
                entry["why_not"] = f"target overlaps #{index} by only {iou:.2f}"
                continue
            fitted, scale = fit_box_to_target(boxes[index], camera, target)
            if fitted is None:
                entry["why_not"] = str(scale)
                continue
            heading = finding.get("heading", "unchanged")
            yaw = HEADING_YAW.get(heading)
            if yaw is not None:
                fitted[box_schema.YAW] = yaw
            before = aabb_centre(boxes[index], camera)
            after = aabb_centre(scale_about_bottom(boxes[index], scale), camera)
            landed = aabb_centre(fitted, camera)
            if landed is None or after is None or before is None:
                entry["why_not"] = "fitted box does not project into the frame"
                continue
            entry.update({
                "applied": True,
                "anchor": boxes[index][box_schema.CENTER].tolist(),
                "box": index,
                "fitted": fitted,
                "name": str(finding.get("name") or (names[index] if index < len(names)
                                                    else "car")),
                # The shift is stored net of the resize, so replaying it on
                # another frame's box (scale, then yaw, then shift) reproduces
                # this fit rather than double-counting what the scale moved.
                "transform": {"scale": float(scale), "yaw": yaw,
                              "shift_px": (landed - after).tolist()},
                "moved_px": float(np.linalg.norm(landed - before)),
            })

        elif op == "add":
            if args.boxes_from == "model":
                fitted, why = box_from_model(finding.get("box3d"))
                if fitted is None:
                    entry["why_not"] = why
                    continue
                entry.update({"applied": True, "source": "model",
                              "name": str(finding.get("name") or "car"),
                              "fitted": fitted})
                entry["target_px"] = _projected_aabb(fitted, camera)
                continue
            if target is None:
                entry["why_not"] = "add with no box_2d"
                continue
            rivals = [a for a in current.values() if a is not None] + added
            overlap = max((aabb_iou(a, target) for a in rivals), default=0.0)
            if overlap > MAX_ADD_IOU:
                entry["why_not"] = f"already boxed (overlaps another box by {overlap:.2f})"
                continue
            name = str(finding.get("name") or "car")
            foot, seed = size_from_target(target, camera, ground_z, name)
            if foot is None:
                entry["why_not"] = "bottom edge does not meet the ground plane"
                continue
            new = np.zeros(box_schema.BOX_DIM)
            new[box_schema.CENTER] = [foot[0], foot[1], ground_z + seed[2] / 2.0]
            new[box_schema.DIMS] = seed
            new[box_schema.YAW] = HEADING_YAW.get(finding.get("heading"), 0.0)
            fitted, scale = fit_box_to_target(new, camera, target)
            if fitted is None:
                entry["why_not"] = str(scale)
                continue
            added.append(target)
            entry.update({"applied": True, "name": name, "fitted": fitted})

        else:
            entry["why_not"] = f"unknown op {op!r}"

    return decided


def anchor_source(out_dir: Path, data: dict, in_boxes: str = "boxes_3d.json"):
    """The boxes an edit's `at` anchor must be able to find, and whether it differs.

    apply_manual_boxes_3d.py always re-applies onto the lift's own output
    (boxes_3d.lifted.json) rather than onto a file that already carries
    corrections. So on a second pass -- reviewing boxes that a first pass already
    moved -- an anchor taken from what was reviewed points at a position the
    lifted file does not have. Returns the lifted data when that is the case, so
    anchors can be remapped back onto it, and None when the two are the same file.

    Under --in_boxes there is nothing to remap: the file being reviewed is also
    the file the edits are applied to, so an anchor resolved against it already
    names a box the merge will see. Remapping onto the lift would be actively
    wrong there -- it would aim each edit at whatever the lift happened to have
    nearest, which for a box an earlier pass added or moved is a different object.
    """
    if in_boxes != "boxes_3d.json":
        return None
    if not data.get("_manual_3d"):
        return None
    pristine = out_dir / "boxes_3d.lifted.json"
    if not pristine.exists():
        raise SystemExit(
            f"error: {out_dir/'boxes_3d.json'} already carries corrections but\n"
            f"       boxes_3d.lifted.json is gone, so an edit cannot be anchored\n"
            f"       to anything the merge will actually see. Re-run stage 3.")
    return json.loads(pristine.read_text())


def remap_anchor(centre, lifted_entry):
    """A reviewed box's centre -> the nearest lifted box's centre, or None.

    None means the box being edited is not in the lifted file at all, which is
    what a box added by an earlier pass looks like. Editing one of those means
    editing the earlier edit, so it is reported rather than written.
    """
    if lifted_entry is None:
        return np.asarray(centre, dtype=float)
    boxes = box_schema.to_array(lifted_entry.get("boxes", []))
    if not len(boxes):
        return None
    distances = np.linalg.norm(boxes[:, box_schema.CENTER] - np.asarray(centre), axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] > 2 * ANCHOR_RADIUS:
        return None
    return boxes[nearest][box_schema.CENTER]


def review_clip(clip: str, args, reviewer) -> dict:
    """One clip end to end. Returns the report row; writes into out_dir."""
    clip_dir = BASE / "data" / args.dataset / clip
    out_dir = clip_dir / args.run if args.run else clip_dir
    frames_dir = Path(args.frames_dir) if args.frames_dir else clip_dir / "frames"
    boxes_path = out_dir / args.in_boxes
    if not boxes_path.exists():
        return {"clip": clip, "error": f"no {args.in_boxes} in {out_dir}"}

    data = json.loads(boxes_path.read_text())
    lifted = anchor_source(out_dir, data, args.in_boxes)
    if args.from_findings and lifted is not None:
        # The cached answers were given about the LIFTED boxes, and a finding
        # names a box by the number drawn on it. Once an apply has run,
        # boxes_3d.json has different boxes in different slots, so replaying
        # against it silently retargets every finding at whatever now sits at
        # that index -- deleting the wrong box, or reporting a deer as already
        # boxed because the last run added it. Re-derive from what was reviewed.
        data, lifted = lifted, None
        print(f"   re-deriving against {out_dir/'boxes_3d.lifted.json'}, "
              f"the boxes these answers were about", flush=True)
    frames = sorted(k for k in data if not k.startswith("_"))
    notes = load_notes(args, clip)
    chosen = select_frames(frames, args, notes)
    if not chosen:
        return {"clip": clip, "error": "no frames selected"}

    review_dir = out_dir / "gemini_review"
    review_dir.mkdir(exist_ok=True)

    cache = (cached_findings(review_dir, frames, args.in_boxes)
             if args.from_findings else {})
    if args.from_findings:
        if not cache:
            return {"clip": clip, "error": f"no cached findings in {review_dir}"}
        chosen = [fi for fi in cache]           # exactly what was asked before
        print(f"== {clip}: re-deriving from {len(chosen)} cached frame(s), "
              f"no requests", flush=True)
    jobs = [(fi, frames[fi], data[frames[fi]], frames_dir, reviewer,
             notes_for_frame(notes, int(Path(frames[fi]).stem)), args,
             cache.get(fi))
            for fi in chosen]
    if not args.from_findings:
        print(f"== {clip}: {len(chosen)} of {len(frames)} frames "
              f"({'stride ' + str(args.stride) if not args.frames else 'explicit'})",
              flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (fi, result) in enumerate(pool.map(review_one_frame, jobs), 1):
            results[fi] = result
            if result.get("error"):
                if reviewer.halted and reviewer.halted in result["error"]:
                    continue          # said once, by the summary below
                print(f"   {frames[fi]}  ERROR {result['error']}", flush=True)
            else:
                hits = sum(1 for d in result["decided"] if d["applied"])
                if hits:
                    print(f"   {frames[fi]}  {hits} edit(s): " +
                          "; ".join(d["reason"][:60] for d in result["decided"]
                                    if d["applied"]), flush=True)
            if done % 10 == 0:
                print(f"   ... {done}/{len(jobs)}", flush=True)

    if reviewer.halted:
        print(f"   {reviewer.halted}", flush=True)
    reviewed = sum(1 for r in results.values() if not r.get("error"))
    if not reviewed:
        return {"clip": clip, "error": "no frame was reviewed successfully"}
    if reviewed < len(chosen):
        print(f"   {len(chosen) - reviewed} of {len(chosen)} frames failed; "
              f"the edits below come from the {reviewed} that did not", flush=True)

    edits, stats = build_edits(frames, data, results, lifted, args)
    write_outputs(out_dir, review_dir, frames, results, edits, stats, args)
    return {"clip": clip, "frames_reviewed": len(chosen), "results": results,
            "edits": edits, "stats": stats, "out_dir": out_dir}


def build_edits(frames, data, results, lifted, args):
    """Accepted corrections -> manual_boxes_3d.json edits, propagated along tracks."""
    stats = {"delete": 0, "replace": 0, "add": 0, "propagated": 0, "unanchored": 0,
             "findings": 0, "frames": 0}
    deletes = {}                      # frame index -> {slot}
    replaces = {}                     # frame index -> {slot: box row}
    adds, add_names = {}, {}

    for fi, result in results.items():
        if result.get("error"):
            continue
        for order, entry in enumerate(result["decided"]):
            if not entry["applied"]:
                continue
            stats["findings"] += 1
            if entry["op"] == "delete":
                deletes.setdefault(fi, {})[entry["box"]] = entry
            elif entry["op"] == "fix":
                replaces.setdefault(fi, {})[entry["box"]] = entry
            elif entry["op"] == "add":
                adds[(fi, order)] = entry["fitted"]
                add_names[(fi, order)] = entry["name"]

    # --- push each correction along its object's track -------------------------
    resolved_delete = {fi: dict(slots) for fi, slots in deletes.items()}
    resolved_replace = {fi: {s: e["fitted"] for s, e in slots.items()}
                        for fi, slots in replaces.items()}
    labels = {}                       # (frame index, slot) -> the finding it came from

    if args.propagate == "track" and (deletes or replaces):
        of_box, members = track_index(frames, data, args.max_dist, args.max_gap,
                                      args.max_px)
        by_track = {}
        for fi, slots in deletes.items():
            for slot, entry in slots.items():
                by_track.setdefault(of_box.get((fi, slot)), {}) \
                        .setdefault("delete", []).append((fi, entry))
        for fi, slots in replaces.items():
            for slot, entry in slots.items():
                by_track.setdefault(of_box.get((fi, slot)), {}) \
                        .setdefault("fix", []).append((fi, entry))

        for ti, found in by_track.items():
            if ti is None:
                continue
            for gj, slot in members[ti]:
                if "delete" in found:
                    # A box that is on a hedge on one frame is on a hedge on all
                    # of them, and a track carrying a delete is dropped whole --
                    # any fix found on the same track goes with it.
                    if slot not in resolved_delete.get(gj, {}):
                        resolved_delete.setdefault(gj, {})[slot] = found["delete"][0][1]
                        stats["propagated"] += 1
                    resolved_replace.get(gj, {}).pop(slot, None)
                    continue
                if slot in resolved_replace.get(gj, {}):
                    labels[(gj, slot)] = None
                    continue
                # Several reviewed frames on one track split it at the midpoints:
                # each frame takes the correction from the frame nearest to it,
                # which is the one whose view of the object it most resembles.
                fi, entry = min(found["fix"], key=lambda pair: abs(pair[0] - gj))
                if entry.get("transform") is None:
                    continue          # placed outright; it means only its own frame
                row = box_schema.to_array(data[frames[gj]]["boxes"])[slot]
                moved = apply_transform(row, entry["transform"], data[frames[gj]]["camera"])
                resolved_replace.setdefault(gj, {})[slot] = moved
                labels[(gj, slot)] = entry
                stats["propagated"] += 1

    # --- and fill the gaps between added keyframes ----------------------------
    resolved_add = {}
    for (fi, order), row in adds.items():
        resolved_add.setdefault(fi, []).append((row, add_names[(fi, order)]))
    if args.propagate == "track" and adds:
        for chain in link_adds(adds, add_names, args.max_add_gap):
            name = chain[0][2]
            for fi, row in interpolate_chain(chain, range(len(frames))):
                resolved_add.setdefault(fi, []).append((row, name))
                stats["propagated"] += 1

    # --- write them out -------------------------------------------------------
    edits = {}
    for fi in sorted(set(resolved_delete) | set(resolved_replace) | set(resolved_add)):
        frame = frames[fi]
        entry = data[frame]
        rows = box_schema.to_array(entry.get("boxes", []))
        names = list(entry.get("names", []))
        scores = list(entry.get("scores", []))
        lifted_entry = lifted.get(frame) if lifted else None
        out = []

        for slot in sorted(resolved_delete.get(fi, {})):
            anchor = remap_anchor(rows[slot][box_schema.CENTER], lifted_entry)
            if anchor is None:
                stats["unanchored"] += 1
                continue
            out.append(make_edit("delete", anchor=anchor,
                                 why=resolved_delete[fi][slot]["reason"]))
            stats["delete"] += 1

        for slot, row in sorted(resolved_replace.get(fi, {}).items()):
            anchor = remap_anchor(rows[slot][box_schema.CENTER], lifted_entry)
            if anchor is None:
                stats["unanchored"] += 1
                continue
            source = labels.get((fi, slot)) or \
                replaces.get(fi, {}).get(slot, {})
            out.append(make_edit(
                "replace", anchor=anchor, box=row,
                name=source.get("name") or (names[slot] if slot < len(names) else "car"),
                score=scores[slot] if slot < len(scores) else 1.0,
                why=source.get("reason", "")))
            stats["replace"] += 1

        for row, name in resolved_add.get(fi, []):
            out.append(make_edit("add", box=row, name=name, score=1.0,
                                 why="added by review"))
            stats["add"] += 1

        if out:
            edits[frame] = out
    stats["frames"] = len(edits)
    return edits, stats


# ------------------------------------------------------------------- writing --
def jsonable(value):
    """numpy out, plain Python in -- the findings file is read by people."""
    if isinstance(value, np.ndarray):
        return [round(float(v), 4) for v in value.ravel()]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def merge_manual(path: Path, edits: dict, in_boxes: str = "boxes_3d.json") -> dict:
    """This run's edits into manual_boxes_3d.json, keeping anything hand-made.

    Only edits tagged "by": "gemini" are replaced. A file that also holds boxes
    someone drew in annotate_boxes.py --boxes3d keeps them, and a second review
    pass does not stack a second copy of its own findings on top of the first.
    """
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    kept = {}
    for frame, frame_edits in existing.get("edits", {}).items():
        hand = [e for e in frame_edits if e.get("by") != "gemini"]
        if hand:
            kept[frame] = hand
    for frame, frame_edits in edits.items():
        kept.setdefault(frame, []).extend(frame_edits)
    merged = dict(existing)
    # What these anchors were resolved against. apply_manual_boxes_3d.py applies
    # onto the lift's own boxes, so edits anchored to anything else would land on
    # whatever the lift happened to have nearest -- it refuses rather than guess.
    if in_boxes != "boxes_3d.json":
        merged["_anchored_to"] = in_boxes
    else:
        merged.pop("_anchored_to", None)
    merged["edits"] = {k: kept[k] for k in sorted(kept)}
    return merged


def write_outputs(out_dir, review_dir, frames, results, edits, stats, args):
    findings = {}
    for fi, result in sorted(results.items()):
        frame = frames[fi]
        if result.get("error"):
            findings[frame] = {"error": result["error"]}
            continue
        decided = result["decided"]
        findings[frame] = {"summary": result["summary"],
                           "findings": [jsonable(d) for d in decided]}
        hits = [d for d in decided if d.get("target_px")]
        want = args.save_images == "all" or (args.save_images == "findings" and decided)
        if want:
            image = annotate_findings(result["image"], hits, result["camera"]["image_hw"]) \
                if hits else result["image"]
            cv2.imwrite(str(review_dir / f"review_{Path(frame).stem}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Which boxes these answers are about; cached_findings refuses to replay them
    # against anything else. Not a frame key, and skipped by the loader's filter.
    findings["_reviewed"] = args.in_boxes
    (review_dir / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    (review_dir / "report.md").write_text(build_report(frames, results, stats, args))

    _tally_head = (f"{stats['findings']} finding(s) -> {stats['delete']} delete, "
                   f"{stats['replace']} replace, {stats['add']} add over "
                   f"{stats['frames']} frame(s) "
                   f"({stats['propagated']} propagated along tracks)")
    tally = _tally_head
    if args.dry_run:
        print(f"   dry run: {tally}; nothing written")
        return
    manual_path = out_dir / MANUAL
    manual_path.write_text(
        json.dumps(merge_manual(manual_path, edits, args.in_boxes), indent=2) + "\n")
    print(f"   wrote {manual_path}: {tally}")
    if args.apply:
        apply_to_boxes(out_dir, edits, args.out_boxes, args.in_boxes)


def apply_to_boxes(out_dir: Path, edits: dict, out_name: str,
                   in_boxes: str = "boxes_3d.json"):
    """Write a full boxes_3d.json carrying this run's corrections, as `out_name`.

    The result is a complete file in the same shape as boxes_3d.json -- every
    frame, every field -- with the corrections merged in, written beside it
    rather than over it. So the lift's own output stays the lift's own output,
    the two can be rendered side by side, and a review that turns out badly
    costs a delete rather than a re-lift.

    Runs apply_manual_boxes_3d.py's own apply_edits rather than a second
    implementation of it, so this and running that script by hand cannot drift
    apart.

    Only `boxes` and the lists that must stay parallel to it -- `names`,
    `scores`, `manual_tracks` -- are touched. Those three are not optional
    company: smooth_boxes and the navsim export index them together with
    `boxes`, so writing a box without a name is a file that raises or silently
    mislabels the box after it. `lanes`, `traffic_lights` and `camera` are left
    exactly as the lift wrote them.
    """
    import apply_manual_boxes_3d as apply3d

    if in_boxes != apply3d.MERGED:
        # Reviewing a corrected file: apply onto that same file, because that is
        # what every anchor was resolved against. Going back to the lift here
        # would silently discard the corrections being reviewed.
        boxes_by_frame = json.loads((out_dir / in_boxes).read_text())
    else:
        source_path = out_dir / apply3d.MERGED
        pristine_path = out_dir / apply3d.PRISTINE
        # Always start from the lift's own boxes: the edits were resolved against
        # them, and their anchors mean nothing in a file that already carries
        # corrections. Which file that is depends on whether anything has ever been
        # applied in place here.
        current = json.loads(source_path.read_text())
        if current.get("_manual_3d"):
            if not pristine_path.exists():
                raise SystemExit(
                    f"error: {source_path} holds in-place corrections but "
                    f"{apply3d.PRISTINE} is gone; re-run stage 3.")
            boxes_by_frame = json.loads(pristine_path.read_text())
        else:
            boxes_by_frame = current
    boxes_by_frame.pop("_manual_3d", None)

    applied, unmatched = apply3d.apply_edits(boxes_by_frame, edits)
    boxes_by_frame["_manual_3d"] = True
    out_path = out_dir / out_name
    tmp = out_path.with_suffix(".json.tmp")
    box_schema.dump(boxes_by_frame, tmp)
    tmp.replace(out_path)
    print(f"   wrote {out_path.name}: {applied['delete']} deleted, "
          f"{applied['replace']} replaced, {applied['add']} added"
          + (f"; {len(unmatched)} anchor(s) matched nothing" if unmatched else ""))
    print(f"   {in_boxes} itself is untouched." if out_path.name != in_boxes
          else f"   {in_boxes} was reviewed and overwritten in place.")
    print(f"   Render it with:")
    print(f"     cd visualization && python raster_frames.py --output_dir {out_dir} "
          f"--frames_dir {out_dir.parent}/frames \\\n"
          f"         --boxes {out_name} --vis_subdir gemini_vis3d")


def build_report(frames, results, stats, args) -> str:
    lines = ["# Gemini box review", "",
             f"model `{args.model}`, propagate `{args.propagate}`, "
             f"min confidence {args.min_confidence}"
             + (f", asked to box: {', '.join(args.find)}" if args.find else ""), "",
             f"{stats['findings']} accepted finding(s) became {stats['delete']} delete, "
             f"{stats['replace']} replace and {stats['add']} add edits over "
             f"{stats['frames']} frame(s); {stats['propagated']} of those were "
             f"propagated along a track from a frame that was actually reviewed."]
    if stats["unanchored"]:
        lines.append(f"{stats['unanchored']} edit(s) could not be anchored to the "
                     f"lifted boxes and were dropped -- they sit on boxes an earlier "
                     f"pass added, so they are edits to an edit.")
    lines += ["", "Nothing here has been applied. Read it, delete what you disagree "
              "with in `manual_boxes_3d.json`, then:", "",
              "```", f"python apply_manual_boxes_3d.py --output_dir <run dir>", "```", ""]

    kept, dropped = [], []
    for fi, result in sorted(results.items()):
        frame = frames[fi]
        if result.get("error"):
            dropped.append(f"| {frame} | - | error | - | {result['error']} |")
            continue
        for entry in result["decided"]:
            # An add has no wireframe to name -- -1 is the sentinel the schema
            # asks for, not a box number, and printing it as one reads as a bug.
            which = f"#{entry['box']}" if entry["box"] >= 0 else "new"
            row = (f"| {frame} | {which} | {entry['op']} | "
                   f"{entry['confidence']:.2f} | {entry['reason']} |")
            (kept if entry["applied"] else dropped).append(
                row if entry["applied"]
                else row[:-1] + f" _(dropped: {entry['why_not']})_ |")

    for title, rows in (("Accepted", kept), ("Not applied", dropped)):
        lines += [f"## {title} ({len(rows)})", "",
                  "| frame | box | op | conf | what |", "|---|---|---|---|---|"]
        lines += rows or ["| - | - | - | - | nothing |"]
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="CARE_YTB", help="dataset dir under data/")
    ap.add_argument("--clip", nargs="*", default=None, help="clip name(s)")
    ap.add_argument("--all", action="store_true", help="every clip in the dataset")
    ap.add_argument("--run", default="1", help="run subdir holding boxes_3d.json ('' = clip dir)")
    ap.add_argument("--frames_dir", default=None, help="override the source frames dir")

    ap.add_argument("--stride", type=int, default=5,
                    help="review every Nth frame (default 5)")
    ap.add_argument("--frames", default=None,
                    help="review exactly these frame numbers, e.g. 70-110,150")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the frames reviewed per clip; 0 = no cap")

    ap.add_argument("--note", action="append", default=[],
                    help="a mistake you already found. May start 'frames 40-90:' to "
                         "scope it, which also forces those frames into the review. "
                         "Repeatable.")
    ap.add_argument("--find", action="append", default=[],
                    help="also box every object of this class, whether or not the "
                         "detector has a label for it: --find deer. Repeatable. "
                         "Added boxes are seeded from the class's real proportions "
                         "rescaled to this clip, since there is no example to copy.")
    ap.add_argument("--notes_file", default=None,
                    help="a file of notes, one per line, optionally grouped under "
                         "[clip] headings")

    ap.add_argument("--model", default="gemini-3.7-flash",
                    help="the 3.x preview Flash models carry the harshest free "
                         "quotas; gemini-2.5-flash is the fallback if this one "
                         "is capped too")
    ap.add_argument("--workers", type=int, default=4, help="frames reviewed in parallel")
    ap.add_argument("--retries", type=int, default=5,
                    help="attempts per frame; a 429's own retry hint is honoured")
    ap.add_argument("--max_output_tokens", type=int, default=4096,
                    help="response budget. Shared with the model's own thinking, so "
                         "too low truncates the JSON and shows up as a parse error "
                         "rather than as a limit (default 4096).")
    ap.add_argument("--thinking", default="low",
                    choices=("minimal", "low", "medium", "high"),
                    help="how much the model may think before answering. Thinking is "
                         "billed as output, and this is a perception task rather than "
                         "a reasoning one (default low).")
    ap.add_argument("--rpm", type=float, default=0.0,
                    help="hold the whole run under this many requests per minute. "
                         "The free API tier allows 5, at which --workers above 1 "
                         "only buys 429s; 0 (default) does not throttle.")
    ap.add_argument("--boxes_from", choices=("fit", "model"), default="fit",
                    help="fit (default): the reviewer returns box_2d and the 3D box "
                         "is solved from it -- back-projected onto the road and "
                         "sized by pixel height, through the clip's own camera. "
                         "model: the reviewer returns box3d and it is used as given, "
                         "ignoring box_2d entirely. model trusts a language model "
                         "with metric depth; fit measures it.")
    ap.add_argument("--ops", default=None,
                    help="restrict what the run may do, e.g. --ops add,delete. "
                         "A finding of any other kind is reported and dropped. "
                         "Use it to keep a targeted pass targeted.")
    ap.add_argument("--min_confidence", type=float, default=0.5,
                    help="findings below this are reported but not applied")

    ap.add_argument("--propagate", choices=("track", "frame"), default="track",
                    help="track: push each correction along the object's whole track "
                         "(default). frame: edit only the frames actually reviewed.")
    ap.add_argument("--max_dist", type=float, default=3.0, help="association gate, m")
    ap.add_argument("--max_px", type=float, default=400.0, help="association gate, px")
    ap.add_argument("--max_gap", type=int, default=2, help="frames a track may miss")
    ap.add_argument("--max_add_gap", type=int, default=10,
                    help="furthest apart two added boxes may be and still be joined "
                         "into one track whose gap is interpolated (default 10). "
                         "Past this the added frames stay isolated rather than "
                         "inventing the motion between them, so a --stride coarser "
                         "than this fills no gaps -- which is correct, nothing was "
                         "seen there.")

    ap.add_argument("--save_images", choices=("all", "findings", "none"),
                    default="findings", help="which review images to keep")
    ap.add_argument("--system_file", default=None,
                    help="file holding the review rules, replacing the built-in "
                         "system instruction. scripts/vis3d/run_gemini.sh keeps "
                         "its copy inline as a heredoc and passes it here.")
    ap.add_argument("--from_findings", action="store_true",
                    help="re-derive the edits from the cached findings.json of a "
                         "previous run instead of asking again. Costs no requests; "
                         "use it to re-tune --max_add_gap, --propagate, --ops or "
                         "--min_confidence against answers you already paid for.")
    ap.add_argument("--apply", action="store_true",
                    help="also write a complete corrected boxes file (see "
                         "--out_boxes), built by apply_manual_boxes_3d.py's own "
                         "apply_edits. boxes_3d.json is never modified.")
    ap.add_argument("--in_boxes", default="boxes_3d.json",
                    help="name of the boxes file to REVIEW, in the run dir. The "
                         "default is the lift's own output. Point it at a "
                         "corrected file (gemini_boxes_3d.json) to review the "
                         "boxes as they now stand rather than as they were "
                         "lifted -- edits are then anchored to, and applied "
                         "onto, that file. Pass the same name as --out_boxes to "
                         "correct it in place.")
    ap.add_argument("--out_boxes", default="gemini_boxes_3d.json",
                    help="name of the corrected boxes file --apply writes, beside "
                         "boxes_3d.json (default gemini_boxes_3d.json)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="report only; write no manual_boxes_3d.json")
    args = ap.parse_args()
    args.ops = {o.strip() for o in args.ops.split(",")} if args.ops else None

    dataset_dir = BASE / "data" / args.dataset
    if args.all:
        clips = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    elif args.clip:
        clips = list(args.clip)
    else:
        raise SystemExit("error: pass --clip NAME [NAME ...] or --all")

    reviewer = (types.SimpleNamespace(halted="") if args.from_findings
                else Reviewer(args.model, args.retries, args.rpm,
                              load_system(args.system_file) if args.system_file
                              else SYSTEM_INSTRUCTION,
                              args.max_output_tokens, args.thinking,
                              args.boxes_from))
    # Halting is per run, not per clip: a quota that ran out on clip 3 has not
    # come back by clip 4, and --all would otherwise walk 50 clips to say so.
    failures = []
    for clip in clips:
        try:
            row = review_clip(clip, args, reviewer)
        except Exception as exc:          # noqa: BLE001 -- one clip must not stop the rest
            row = {"clip": clip, "error": str(exc)}
        if row.get("error"):
            print(f"== {clip}: {row['error']}")
            failures.append(clip)
            if reviewer.halted:
                print(f"   not attempting the remaining clips")
                break
        else:
            print(f"   report: {row['out_dir']}/gemini_review/report.md")
        print()

    if failures:
        print(f"{len(failures)} clip(s) failed: {' '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
