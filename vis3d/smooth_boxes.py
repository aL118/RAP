#!/usr/bin/env python3
"""Temporally smooths lift_frames_to_3d.py's boxes_3d.json.

Every frame is fit independently, so a parked car's box wanders by ~0.5 m and
breathes by ~10% in length from one frame to the next, and the rasterizer's
per-face palette makes that read as colour flicker: as the box tilts, the top
face (purple) grows or vanishes and the depth attenuation shifts.

Boxes carry no track ids, so this associates them across frames first (greedy
nearest-centroid within a class), then runs a Gaussian filter along time over
each track's centre, extent and heading.

Association is scale-relative in both of its terms, because the objects a
dashcam sees span two decades of apparent size within a single frame. The
distance gate is a fraction of the object's own projected size rather than a
fixed number of pixels, and a size-ratio test rejects pairs whose on-screen
extents cannot belong to the same object however close their centres are. See
associate() for what a single absolute gate does to a truck at 12 m.

Association also decides which boxes are objects at all, in both directions: a
track seen in one or two frames of a clip is a detector flicker, not a vehicle,
and --min_track_len deletes it everywhere it appeared; a track that goes missing
for a frame or three in the middle is the same detector losing an object it is
otherwise sure of, and --max_fill interpolates the hole shut. Either way a box
stops blinking, which is what the eye picks up in a video long before it
notices where the box actually is.

Heading is smoothed on the doubled angle. `_fit_box` reads it off the longer
minAreaRect edge, which is only defined up to +-pi, so exp(2i*yaw) is the
quantity that is actually continuous; averaging raw yaw would drag a box
through a spurious half-turn whenever the representative flips.

A median can only remove what comes back, though, and the worst of what is left
does not: the lifter's heading search picks between 36 candidate yaws on a
likelihood barely above the noise, so a moment of bad mask hands the argmax to a
rival 20-45 deg away and every frame after it agrees. So heading and extent are
finally rate-limited -- a box may not turn or change size faster than a real one
could between two frames. See MAX_YAW_RATE_DEG and MAX_SIZE_RATE for the measured
gap the thresholds sit in, and rate_limit_angle for why a flip has to be dropped
rather than slowed down.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import box_schema
# The size priors lift_frames_to_3d.py built each box from, needed to undo a
# per-frame class label and re-impose the track's settled one.
from lift_frames_to_3d import CLASS_LENGTH_PRIOR, CLASS_SIZE_PRIOR
# The height the renderer draws a light at, which is also what its range was
# fitted through -- so it converts back to the apparent size the gate needs.
from traffic_lights import TRAFFIC_LIGHT_DIMS

# Rows are worked on as the (9,) array box_schema.from_any returns, whatever
# shape they had in the file; only the write at the end turns them back into
# named-field dicts. This filter estimates a yaw and leaves roll and pitch as
# it found them.
POS, DIM, YAW = box_schema.CENTER, box_schema.DIMS, box_schema.YAW

# Association gate, as a fraction of the object's own projected size (the
# geometric mean of its silhouette's pixel width and height).
#
# A single absolute pixel gate cannot serve this footage. Measured on the
# beepbeep clip: a truck's projected centre moved 127 px between frames 18 and
# 19 -- 17% of its own 767 px width, unmistakably the same object -- while for a
# 40 px distant car the same 127 px is three times its whole extent, and
# certainly a different one. Under the old fixed 120 px gate the truck's track
# broke exactly there, its box was greedily adopted by a track of small cars 20 m
# away whose centre happened to lie 25 px off, and the orphaned track then
# interpolated a second 7.5 m box across the hole it thought it had -- two boxes
# on one truck.
#
# 0.5 admits every same-object pair measured on that clip (the largest genuine
# centre movement is 0.35 of the object's size) with a wide margin, while still
# holding a small distant car to a few tens of pixels.
GATE_EXTENT_FRAC = 0.5

# Floor under the scaled gate, in pixels. A car at 40 m projects to ~30 px and
# would otherwise be gated at 15 px, which is inside the jitter of the box fit
# itself rather than any real motion.
MIN_GATE_PX = 40.0

# Largest ratio between two projected sizes still taken to be one object.
#
# This is the test that stops identity theft between objects at different
# ranges, which distance alone cannot see: the truck above is 464 px across and
# the car whose track adopted it is 48 px, a ratio of 9.7 at a centre distance
# of 25 px.
#
# 2.5 is set from the same-object distribution: median ratio 1.03, 90th
# percentile 1.43, and 96% of true pairs below 2.0. The tail above it is a
# vehicle looming on a collision course, whose silhouette can genuinely grow by
# 1.6x in a frame; 2.5 clears that with room to spare. Rejecting a true pair
# costs a track split, which --max_fill and --min_track_len are already there to
# absorb; accepting a false one puts a box on the wrong object.
MAX_SIZE_RATIO = 2.5

# Weight on the size-mismatch term in the association cost, in octaves of size
# ratio per unit of gate-normalised distance. At 1.0 a candidate twice the size
# of the track it is matching is penalised as much as one sitting a full gate
# width away, so where two candidates are both admissible the one that is the
# right size wins over the one that is merely closer.
SIZE_COST_WEIGHT = 1.0

# Smallest fraction of its silhouette a box must have inside the image before its
# class label is allowed into the track's vote.
#
# Apparent size is a good proxy for how readable a detection is, but only up to
# the point where the object stops fitting in frame. Past that it is the wrong
# way round, and on a dashcam that region is where the interesting vehicles live.
# The beepbeep truck: at 12-16 m it is a whole truck, fully in frame, and the
# detector reads it as truck, truck, truck, bus. From frame 21 it is close enough
# that only a slab of cab door is in shot -- 64% of its silhouette, then 45%,
# then 18% -- and every one of those frames comes back "car", because a
# featureless painted panel is what a car looks like too.
#
# Weighting by size alone hands the vote to exactly those frames: they are the
# largest ones. So the vote is taken over the untruncated views if the track has
# any, and only falls back to the clipped ones when it has none.
#
# 0.9 rather than 1.0 because a box's silhouette is the whole cuboid, which flares
# a little past the mask it was anchored to, so a vehicle sitting right at the
# border can lose a few percent without being meaningfully cut off.
VOTE_MIN_VISIBLE = 0.9

# Largest heading change a box may make between two frames, in degrees.
#
# What survives robust_angle_filter is not noise, it is a mode flip. The lifter
# picks a heading by scoring 36 candidate yaws against the mask silhouette, and
# the IoU spread across a whole half-turn is only about 0.1 -- comparable to the
# roughness of the mask itself (see lift_frames_to_3d.HEADING_PRIOR_WEIGHT). So a
# mask that changes shape for a moment can hand the argmax to a rival candidate
# 20-45 deg away, and then go on handing it there for the rest of the track. A
# median window cannot touch that: two frames after the flip the wrong mode owns
# the majority of every window it appears in.
#
# The flip is separable from real motion in the derivative, not in the value.
# Measured on wrongway/4: subtract the scene-wide rotation (which is the ego's,
# and which every static object shares) and every track's heading is constant to
# 0.69 deg/frame at the 90th percentile and 1.3 at the 95th, while every flip
# moves at least 11.3 deg in a single frame. Two orders of magnitude apart, with
# 5.7 deg/frame the largest step anywhere in between.
#
# 8 deg/frame sits in that gap. At the 10 Hz this pipeline processes at that is
# 80 deg/s, more than a vehicle and the ego could turn against each other at
# once, and it still rejects the smallest observed flip with 30% to spare.
MAX_YAW_RATE_DEG = 8.0

# Largest change in a box's *angular* size between two frames, as a fraction.
#
# Angular size, not metric: what pulses on screen is extent over depth, and the
# lifter scales the two together so the projection stays put -- the same reason
# the extent median filters dims/depth rather than dims.
#
# Real growth is bounded by how fast the gap closes: d(log size) = closing speed
# / range, so a car closing at 15 m/s from 5 m -- three tenths of a second from
# contact -- grows 30% in a frame at 10 Hz, and everything less extreme grows far
# less. The measured distribution agrees and then stops: on wrongway/4 the 95th
# percentile step is 11%, the 98th is 29%, and the next value up is 44%. 0.35
# sits in that gap, clipping the 1.8% of steps where a mask spilled onto a
# neighbour or collapsed onto a highlight, and nothing else.
MAX_SIZE_RATE = 0.35

# Half-width of the deadband each damped quantity is held inside, as a fraction
# (or, for the image centre, in pixels).
#
# The rate limits above answer "no vehicle could have changed that much between
# two frames". These answer the opposite question: a vehicle does not jiggle. A
# real box's range, size and position each follow a trajectory -- monotone over a
# second or two, because that is what driving looks like -- and anything that
# leaves one and comes straight back is the estimator, not the object.
#
# Reversal rate is what separates the two, and it is unambiguous here. Measured
# on wrongway/4, the fraction of consecutive steps that reverse direction:
#
#   range (depth)        0.46      0.5 is a coin toss, i.e. no trajectory at all
#   image centre         0.24
#   angular size         0.04      after the rate limit; already a trajectory
#   heading              0.04
#
# So the box's range is not being measured, it is being guessed afresh each
# frame: the median frame moves it 6% and the 90th percentile 23%, and the
# residual about its own local trend is 11.6% wide. That is what makes a parked
# car lunge at the camera and fall back, and because the lifter carries extent
# and range together, it is also most of what makes the metric box breathe.
#
# The bands are set to the width of that residual, not to the per-frame step,
# and this is the whole reason a deadband is the right operator rather than a
# low-pass. Per frame the noise (6%) and a genuinely fast approach (a car closing
# at 15 m/s from 20 m covers 7.5% of its range per frame at 10 Hz) are the same
# size and no per-step test can tell them apart. Over five frames they are not:
# the approach has accumulated 30% and the noise is still inside its 12% band.
# Holding until the measurement leaves the band is exactly that test.
DAMPEN_RANGE = 0.12          # residual sd 11.6% of range
DAMPEN_SIZE = 0.03           # residual sd 3.5% of angular size
DAMPEN_CENTRE_PX = 6.0       # residual sd 8.0 px, p90 9.3 px

# --- Range outliers, and why they are an *extent* problem --------------------
#
# A deadband removes wander inside a band; it cannot remove an excursion that
# leaves one and stays out. The ambulance clip has exactly that: over frames
# 56-60 the ambulance's monocular range reads 42-53 m while the frames either
# side read 15 and 25, a 3.6x step in one frame and back five frames later.
#
# What that ruins is not where the box is drawn -- range_scale rescales the
# extent by the same factor, so the projection is untouched and the box stays on
# its object throughout -- but how big the box *is*. A box's metric extent is
# its angular size times its range, and the angular size here is already smooth
# (this track's grows 100 -> 700 px without a step over 30%), so every jump in
# l/w/h is a jump in range wearing a different hat. Across that excursion the
# ambulance's length reads 2.6 m, then 9.2, then 4.6: one vehicle, three
# lengths, none of which it is.
#
# So the fix is the same median-then-trimmed-mean the angular size already gets,
# applied to the range before the deadband sees it. On that track it takes the
# worst per-frame length change from 260% to 28% and leaves the drawn size
# identical to the pixel (p90 step 7%, max 28%, both before and after) -- which
# is the point: nothing about the picture changes, only the metres written into
# the log. The window is --size_window, a duration like every other window here;
# the band is its own because range noise (12%) is four times the angular size's.
RANGE_INLIER_FRAC = 0.25

# Last word on the extent, after the range rescale. What survives the range
# filter is slow drift rather than jumps -- a rigid vehicle whose logged length
# still creeps because range and angular size disagree about how fast it is
# approaching -- and a deadband is the operator for that: it holds l/w/h still
# until the evidence leaves the band, so a box that should be one size stays one
# size. Applied to the metric extent and therefore the one filter here that does
# move the projection, which is why the band is loose: at 0.10 it changes the
# drawn size by less than the range deadband already does.
DAMPEN_EXTENT = 0.10

# --- Traffic lights ---------------------------------------------------------
#
# Lights get the same treatment as boxes and for the same reasons, but they were
# passed through untouched until now: the lifter wrote them per frame and nothing
# downstream ever looked across frames. Both failures that causes are visible on
# the ambulance clip. A light detected in frames 35, 40 and 43 and in none of the
# ones between blinks three times in half a second, because there was no track to
# say it had been there all along. And its state was decided afresh every frame
# from a mask a few dozen pixels wide, so a light that is one colour for four
# seconds got re-argued 40 times over.
#
# The state vote is what makes traffic_lights.classify_state able to abstain.
# Read frame by frame that reader is deliberately cautious -- it answers
# "unknown" wherever the lamp is not clearly the compact blob it is looking for,
# which on that clip is 92 of 142 detections. Over a track those abstentions cost
# nothing: 50 decided frames carry the other 92, and the light is drawn one
# colour throughout instead of flickering between a guess and white.
#
# The gate is a fraction of the light's own apparent size, clamped, exactly as
# gate_px does it for vehicles -- a light overhead sweeps across the frame as the
# ego reaches the stop line, and a fixed pixel gate either severs that or is wide
# enough to swap two lights on the same mast. Measured over the clip's adjacent
# frames, a light's projected centre moves a median 0.23 of its own long axis.
TRAFFIC_LIGHT_GATE_FRAC = 1.0
TRAFFIC_LIGHT_MIN_GATE_PX = 60.0
TRAFFIC_LIGHT_MAX_GATE_PX = 250.0


def _write(out: dict, dst: Path) -> None:
    """Serialise the filtered frames back to boxes_3d.json.

    Rows are lists in here; the file wants named fields. Going through
    box_schema means this stage cannot undo the lifter's formatting, which is
    what used to happen -- the lifter wrote indent=2 and this wrote one long
    line over the top of it.
    """
    out = {frame: {**entry, "boxes": box_schema.to_dicts(entry.get("boxes", []))}
           for frame, entry in out.items()}
    box_schema.dump(out, dst)


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def smooth_1d(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolves along axis 0, holding the end values constant outside the track.

    Edge padding rather than zero padding: a track's first and last samples are
    real measurements, and zero-padding would pull them toward the origin.
    """
    if len(values) == 1:
        return values.copy()
    pad = len(kernel) // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(values, dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def centre_depths(entry):
    """Camera-frame depth of each box centre, in metres."""
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    return [float((transform[:3, :3] @ box_schema.from_any(b)[POS]
                   + transform[:3, 3])[2]) for b in entry.get("boxes", [])]


def projected_centres(entry):
    """Each box's centre projected to pixels, or None if it is behind the camera."""
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(entry["camera"]["intrinsics"], dtype=np.float64)
    out = []
    for box in entry.get("boxes", []):
        camera = transform[:3, :3] @ box_schema.from_any(box)[POS] + transform[:3, 3]
        out.append(None if camera[2] <= 1e-3 else (intrinsics @ camera)[:2] / camera[2])
    return out


def projected_bounds(entry, rows=None):
    """Per box: (top-left, bottom-right) of its projected silhouette, or None.

    None when any corner is behind the camera, where the projection is not a
    rectangle at all. `rows` defaults to the entry's own boxes; a caller holding
    boxes that are not in the entry yet -- the gap filler, deciding whether to
    insert one -- passes them in.
    """
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(entry["camera"]["intrinsics"], dtype=np.float64)
    signs = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
                      [1, 1, -1, -1, 1, 1, -1, -1],
                      [1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float64)
    out = []
    for box in (entry.get("boxes", []) if rows is None else rows):
        row = box_schema.from_any(box)
        length, width, height = row[DIM]
        cos, sin = np.cos(row[YAW]), np.sin(row[YAW])
        local = signs * np.array([[length / 2], [width / 2], [height / 2]])
        corners = np.stack([cos * local[0] - sin * local[1] + row[POS][0],
                            sin * local[0] + cos * local[1] + row[POS][1],
                            local[2] + row[POS][2]])
        camera = transform[:3, :3] @ corners + transform[:3, 3:4]
        if (camera[2] <= 1e-3).any():
            out.append(None)
            continue
        uv = (intrinsics @ camera)[:2] / camera[2]
        out.append((uv.min(axis=1), uv.max(axis=1)))
    return out


def projected_silhouettes(entry):
    """Per box: (apparent size, visible area, visible fraction), or None.

    `size` is the geometric mean of the projected silhouette's pixel width and
    height, so one number stands for "how big this object is on screen"
    regardless of whether it is seen end-on or broadside. That is the quantity
    association has to be relative to: a box's centre is a position, but only its
    extent says how far that position is allowed to move.

    `visible area` and `visible fraction` measure the silhouette against the
    image rectangle. An object closer than the camera's field of view can hold
    spills over the border, and past that point growing nearer stops adding
    information and starts removing it -- which is why the label vote needs them
    and the geometry does not. None when any corner is behind the camera, which
    every caller reads as "these tests cannot speak here".

    Taken from the box rather than the mask because the mask does not survive
    into boxes_3d.json -- but lift_frames_to_3d.py anchors every box's silhouette
    to its mask's bbox, so the two agree to within the anchoring residual.
    """
    height_px, width_px = entry["camera"]["image_hw"]
    out = []
    for corners in projected_bounds(entry):
        if corners is None:
            out.append(None)
            continue
        low, high = corners
        span = np.maximum(high - low, 1e-6)
        inside = np.maximum(np.minimum(high, [width_px, height_px])
                            - np.maximum(low, 0.0), 0.0)
        out.append((float(np.sqrt(span[0] * span[1])),
                    float(inside[0] * inside[1]),
                    float(inside[0] * inside[1] / (span[0] * span[1]))))
    return out



def size_ratio(size_a, size_b):
    """How many times larger the bigger of two apparent sizes is, or None.

    None when either is unknown (a box straddling the image plane has no
    silhouette), which every caller reads as "this test cannot speak here"
    rather than as a pass or a fail.
    """
    if not size_a or not size_b or size_a <= 0 or size_b <= 0:
        return None
    return max(size_a, size_b) / min(size_a, size_b)


def gate_px(size_a, size_b, frac, floor_px, ceil_px):
    """Distance gate for a pair, as a fraction of the larger apparent size.

    The larger of the two rather than the smaller or their mean: an object that
    is looming, which is when the gate matters most, is bigger at the near end of
    every link it makes, and gating it on the far end's size would tighten the
    gate exactly as its motion grows.
    """
    known = [s for s in (size_a, size_b) if s]
    if not known:
        return float(ceil_px)
    return float(min(max(frac * max(known), floor_px), ceil_px))


def associate(frames, data, max_dist, max_gap, cross_class=False, max_px=None,
              gate_extent_frac=GATE_EXTENT_FRAC, min_px=MIN_GATE_PX,
              max_size_ratio=MAX_SIZE_RATIO, size_weight=SIZE_COST_WEIGHT):
    """Greedy nearest-centroid association, in pixels if `max_px` else in metres."""
    """Returns tracks: list of dicts {idx: [frame_index, ...], box: [row, ...]}.

    `cross_class` matters because the detector relabels an object mid-track on
    about 5% of tracks (car <-> truck). Gating association on the class splits
    such an object into two short tracks exactly where its box changes shape the
    most, so no temporal filter can see across the flip -- which is the one place
    it is needed. Matching across classes keeps the track whole and lets
    majority_name() settle the label afterwards.

    `max_px` gates on the projected centre instead of the 3D one. Boxes are
    anchored to their masks, so where they sit in the image is a measurement,
    while their depth is a monocular guess that can jump several metres between
    consecutive frames -- enough to blow a metric gate and split a track that
    never moved on screen. Observed: a car ahead went 12 m -> 16 m in one frame
    while its mask barely shifted.

    Both terms of the match are relative to the object's own apparent size, and
    neither works without the other. `gate_extent_frac` sets how far a centre may
    move, as a fraction of that size, so the gate travels with a looming vehicle
    instead of severing its track at the moment it fills the frame -- but a gate
    wide enough for a truck is wide enough to swallow every small car near it, so
    on its own it trades a broken track for a stolen one. `max_size_ratio` is
    what makes that safe: two boxes whose silhouettes differ by more than it are
    not the same object at any distance, and `size_weight` puts the residual
    mismatch into the cost so that among admissible candidates the one of the
    right size outranks the one that is merely nearer.

    The cost is the gate-normalised distance, not the raw one. Pairs are resolved
    greedily across the whole frame, so they have to be comparable across objects
    of different sizes; in pixels they are not, and a distant car that barely
    moves would always be matched before a near one that crossed a third of its
    own body.
    """
    tracks, active = [], []           # active: indices into tracks
    for fi, frame in enumerate(frames):
        entry = data[frame]
        boxes = [box_schema.from_any(b) for b in entry.get("boxes", [])]
        names = entry.get("names", [])
        scores = entry.get("scores", [1.0] * len(boxes))
        pixels = projected_centres(entry) if max_px else [None] * len(boxes)
        # Apparent size when gating in pixels; the box's own length when gating
        # in metres. Either way it is only ever read as a ratio between two
        # boxes, so the units cancel and the same tests apply to both paths.
        silhouettes = (projected_silhouettes(entry) if max_px
                       else [None] * len(boxes))
        sizes = ([None if s is None else s[0] for s in silhouettes] if max_px
                 else [float(b[DIM][0]) for b in boxes])
        depths = centre_depths(entry)
        # Padded rather than required: a boxes_3d.json lifted before the field
        # existed has no ids, and every box in it is then a detector box.
        manual = list(entry.get("manual_tracks", []))
        manual += [None] * (len(boxes) - len(manual))

        pairs = []
        for bi, (box, name) in enumerate(zip(boxes, names)):
            for ti in active:
                t = tracks[ti]
                # A hand-drawn box's identity is known, so it is matched on the
                # annotator's track id and skips every gate. It has to: the
                # objects people annotate are the ones the detector lost, which
                # on a dashcam means something closing fast, and a box on such
                # an object crosses hundreds of pixels between frames. Gating it
                # splits one drawn track into fragments and --min_track_len then
                # deletes them, so annotating a frame can leave it emptier than
                # not annotating it at all. Manual and detected boxes never
                # merge in either direction: splicing a detection onto a drawn
                # identity would be a guess wearing a measurement's protection.
                if manual[bi] is not None or t["manual"] is not None:
                    if manual[bi] is not None and t["manual"] == manual[bi]:
                        pairs.append((-1.0, bi, ti))
                    continue
                if not cross_class and t["name"] != name:
                    continue
                # Size first: it is the only test that separates a near truck
                # from a distant hatchback sitting at the same place on screen,
                # and no distance gate wide enough for the first can exclude the
                # second.
                ratio = size_ratio(sizes[bi], t["size"][-1])
                if ratio is not None and ratio > max_size_ratio:
                    continue
                if max_px and pixels[bi] is not None and t["pixel"] is not None:
                    d = float(np.linalg.norm(pixels[bi] - t["pixel"]))
                    gate = gate_px(sizes[bi], t["size"][-1],
                                   gate_extent_frac, min_px, max_px)
                else:
                    d = float(np.linalg.norm(box[POS] - t["box"][-1][POS]))
                    gate = max_dist
                if d > gate:
                    continue
                cost = d / gate
                if ratio is not None:
                    cost += size_weight * float(np.log2(ratio))
                pairs.append((cost, bi, ti))
        pairs.sort(key=lambda p: p[0])

        used_b, used_t = set(), set()
        assigned = {}
        for cost, bi, ti in pairs:
            if bi in used_b or ti in used_t:
                continue
            used_b.add(bi); used_t.add(ti); assigned[bi] = ti

        for bi, (box, name) in enumerate(zip(boxes, names)):
            score = scores[bi] if bi < len(scores) else 1.0
            if bi in assigned:
                t = tracks[assigned[bi]]
            else:
                t = {"name": name, "idx": [], "box": [], "score": [], "slot": [],
                     "names": [], "pixel": None, "size": [], "visible": [],
                     "depth": [], "manual": manual[bi]}
                tracks.append(t)
            t["idx"].append(fi); t["box"].append(box); t["score"].append(score)
            t["names"].append(name)
            t["pixel"] = pixels[bi]
            # Kept per sample rather than as a running last value like `pixel`,
            # because majority_name() weights the label vote by them.
            t["size"].append(sizes[bi])
            t["visible"].append(silhouettes[bi][1:] if silhouettes[bi] else None)
            t["depth"].append(depths[bi])
            # Position within the frame's box list, so a filter that rewrites one
            # field can put it back without rebuilding the entry.
            t["slot"].append(bi)

        # a track stays matchable until it has been missing for max_gap frames.
        # A drawn track is exempt: it is matched by id, and an annotator may
        # leave an object out for a stretch and pick the same track up later.
        active = [ti for ti, t in enumerate(tracks)
                  if t["manual"] is not None or fi - t["idx"][-1] <= max_gap]
    return tracks


def resample(track, fill_gaps):
    """Track samples on a contiguous frame grid, linearly filling short gaps."""
    idx = np.asarray(track["idx"])
    rows = np.stack(track["box"])
    scores = np.asarray(track["score"], dtype=np.float64)
    if not fill_gaps or len(idx) < 2:
        return idx, rows, scores, np.ones(len(idx), dtype=bool)

    grid = np.arange(idx[0], idx[-1] + 1)
    out = np.empty((len(grid), rows.shape[1]), dtype=np.float64)
    for c in range(rows.shape[1]):
        if c == YAW:                       # interpolate the doubled angle
            z = np.interp(grid, idx, np.cos(2 * rows[:, c])) + \
                1j * np.interp(grid, idx, np.sin(2 * rows[:, c]))
            out[:, c] = np.angle(z) / 2.0
        else:
            out[:, c] = np.interp(grid, idx, rows[:, c])
    sc = np.interp(grid, idx, scores)
    observed = np.isin(grid, idx)
    return grid, out, sc, observed


def circular_median(angles: np.ndarray) -> float:
    """The angle minimising total absolute circular distance to `angles`.

    Restricted to the samples themselves, which is what makes it a median rather
    than a mean: on a circle there is no ordering to take a midpoint of, but the
    minimiser of summed absolute deviation is always attained at a sample.
    """
    difference = np.abs(np.angle(np.exp(1j * (angles[:, None] - angles[None, :]))))
    return float(angles[np.argmin(difference.sum(axis=1))])


def robust_angle_filter(yaws, window: int, inlier_rad: float) -> np.ndarray:
    """Median-then-trimmed-mean filter along a track's yaw, on the doubled angle.

    Median, not Gaussian: the artefact here is an isolated frame whose heading
    jumps tens of degrees and comes straight back (15% of frames move >10 deg and
    2% move >30 deg, in tracks that are otherwise steady). A Gaussian smears such
    a spike across its neighbours instead of removing it, and rounds off genuine
    steps as well. A median deletes the spike outright and still follows a real
    step as soon as it owns half the window -- 3 frames at window 5, ~0.1 s at
    30 fps -- so a crash still reads as a crash.

    The trimmed mean afterwards averages only the samples within `inlier_rad` of
    that median. It removes the staircase left by the heading search's discrete
    yaw grid without letting an outlier back in, and at a genuine step it averages
    over one side of the step only.

    Doubled angle throughout: yaw is defined up to +-pi, so exp(2i*yaw) is the
    quantity that is actually continuous across a flip of the representative.
    """
    doubled = 2.0 * np.asarray(yaws, dtype=np.float64)
    out = np.empty_like(doubled)
    half = max(window // 2, 0)
    for i in range(len(doubled)):
        neighbourhood = doubled[max(0, i - half):i + half + 1]
        median = circular_median(neighbourhood)
        offset = np.angle(np.exp(1j * (neighbourhood - median)))
        inliers = neighbourhood[np.abs(offset) <= inlier_rad]
        if len(inliers) == 0:
            out[i] = median
        else:
            out[i] = np.angle(np.mean(np.exp(1j * inliers)))
    return out / 2.0


def rate_limit_angle(yaws, frames, max_step: float, weights=None):
    """Drops heading steps no vehicle could have made, then re-anchors the track.

    Dropped, not clamped, and that is the whole point. A slew limit would still
    walk the box onto a flipped mode, just over ten frames instead of one, because
    the flipped heading is what every frame after the flip reports. Rejecting the
    step instead leaves the estimate on the mode it was already holding, and since
    the post-flip readings are just as steady as the pre-flip ones -- constant to
    a fraction of a degree per frame either side -- it stays there. What is being
    filtered is the derivative: a real turn arrives as a run of small steps and
    passes through untouched, while a mode flip arrives as one impossible step
    and is the only thing this can see.

    The output is therefore the running sum of the surviving steps, which fixes
    the shape of the heading over time but not its offset -- each rejected step
    leaves everything after it displaced by the amount that was dropped. So the
    rejected steps cut the track into segments that differ from the measurements
    by a constant apiece, and `weights` picks which segment's constant to keep:
    the offset is chosen so that the segment with the most weight reads exactly
    as measured. Weighting by apparent area and detection score rather than by
    frame count, for the reason majority_name does -- a heading read off forty
    frames of a distant smudge is not better evidence than one read off eight
    frames where the vehicle fills a third of the image.

    Doubled angle throughout, as everywhere else here: yaw is defined only up to
    +-pi, so a flip of the representative is not a 180 deg step.

    :param frames: frame index of each sample, so a step across a dropout is
        allowed proportionally more movement than a step across one frame
    :param max_step: largest heading change per frame, radians
    :returns: (yaws, number of steps rejected)
    """
    yaws = np.asarray(yaws, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    if len(yaws) < 2 or max_step <= 0:
        return yaws.copy(), 0

    steps = np.angle(np.exp(1j * 2.0 * np.diff(yaws))) / 2.0
    gaps = np.maximum(np.diff(frames), 1)
    kept = np.abs(steps) <= max_step * gaps

    out = np.concatenate([[0.0], np.cumsum(np.where(kept, steps, 0.0))]) + yaws[0]
    # A new segment begins at each rejected step; within one, out - yaws is
    # constant, so any of its samples fixes the offset for the whole segment.
    segment = np.concatenate([[0], np.cumsum(~kept)])
    if weights is None:
        weights = np.ones(len(yaws), dtype=np.float64)
    anchor = int(np.argmax(np.bincount(segment, weights=np.asarray(weights, float))))
    first = int(np.argmax(segment == anchor))
    out -= out[first] - yaws[first]

    # Back to the +-pi/2 representative the rest of this module works in: the
    # running sum is unbounded, and robust_angle_filter's output is not.
    return np.angle(np.exp(2j * out)) / 2.0, int((~kept).sum())


def rate_limit_scale(angular, frames, max_rate: float):
    """Clamps how fast a box's angular size may change between two frames.

    Clamped rather than dropped, the opposite of rate_limit_angle, because the
    two artefacts are not the same shape. A heading has a small set of candidate
    values and the error is landing on the wrong one, which a later frame does
    not correct; a size is a free scalar fitted to the mask, so a bad frame is a
    bad frame and the frames after it are right again. Refusing to follow a real
    change would leave the box permanently off the object it is anchored to,
    while clamping only costs a frame or two of catching up.

    In log space, so the limit is a ratio: a box may grow by `max_rate` or shrink
    by the reciprocal, and neither direction is privileged by where the box
    happens to be in the frame.

    :param angular: (N,) or (N, k) angular sizes, i.e. extent over depth
    :param frames: frame index of each sample, so a step across a dropout is
        allowed proportionally more change
    :returns: (angular sizes, number of frames clamped)
    """
    values = np.asarray(angular, dtype=np.float64)
    if len(values) < 2 or max_rate <= 0:
        return values.copy(), 0

    log = np.log(np.maximum(values, 1e-9))
    gaps = np.maximum(np.diff(np.asarray(frames, dtype=np.int64)), 1)
    limit = np.log1p(max_rate)
    out = log.copy()
    clamped = 0
    for k in range(1, len(log)):
        allowed = limit * gaps[k - 1]
        excess = log[k] - out[k - 1]
        clamped += int(np.any(np.abs(excess) > allowed))
        out[k] = out[k - 1] + np.clip(excess, -allowed, allowed)
    return np.exp(out), clamped


def dampen(values, band: float) -> np.ndarray:
    """Holds an estimate still until the measurement leaves a band around it.

    The deadband, or play, operator -- the backlash in a gear train, and the
    reason a loose steering wheel does not transmit road buzz. The estimate is
    dragged along only by the edge of the band, so a measurement that wanders
    inside it moves the output not at all, while one that keeps going drags the
    output with it indefinitely. Two properties follow, and both are why this is
    the operator rather than another low-pass:

      * any oscillation narrower than the band is removed *entirely*, not
        reduced. Averaging leaves a box that wobbles slightly less; this leaves
        one that is still.
      * the output never departs from the measurement by more than `band`. It
        cannot drift off, cannot ring, and cannot invent a trajectory -- which
        matters because these boxes are anchored to something and the anchor is
        what has to survive.

    Run forwards and backwards and averaged, because one pass alone lags by the
    band in whatever direction it is travelling: the forward sweep trails a
    receding object and the backward sweep leads it by the same amount, so their
    mean sits on the trajectory with no phase error at all. Averaging two outputs
    that are each within `band` of the measurement keeps the bound, and an
    oscillation both sweeps flatten stays flat in the mean.

    Rate-independent, so a track with a dropout in it needs no special case: the
    band is an amplitude, not a speed, and a step across a three-frame gap is
    tested exactly like a step across one. That is the difference between this
    and rate_limit_scale, which is about how fast and knows about gaps.

    :param band: half-width of the deadband, in the units of `values`. Ratios are
        damped by passing their logarithm, so the band is a fraction.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or band <= 0:
        return values.copy()

    def sweep(x):
        out = np.empty_like(x)
        out[0] = x[0]
        for k in range(1, len(x)):
            out[k] = min(max(out[k - 1], x[k] - band), x[k] + band)
        return out

    return 0.5 * (sweep(values) + sweep(values[::-1])[::-1])


def dampen_ratio(values, band: float) -> np.ndarray:
    """dampen() on a quantity whose noise is proportional, so `band` is a fraction.

    Range and size are both of this kind: 12% of 40 m is not 12% of 8 m, and a
    band in metres would be most of a near box and invisible on a far one.
    """
    values = np.asarray(values, dtype=np.float64)
    if band <= 0:
        return values.copy()
    return np.exp(dampen(np.log(np.maximum(values, 1e-9)), np.log1p(band)))


def sight_line(entry, rows):
    """Box centres as (u, v, range): where each sits in the image, and how far off.

    The coordinates the anchoring actually determines, split from the one it does
    not. lift_frames_to_3d.py fits every box to a mask, which fixes where it lies
    in the image and says nothing about how far away it is -- the range comes from
    a monocular point map whose scale wanders. Damping a centre in xyz would blur
    those together and slide the box off its own detection; damping it here moves
    the box along its sight line, where there was never a measurement to preserve.

    :returns: (u, v, range) arrays, or None if any centre is behind the camera
    """
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(entry["camera"]["intrinsics"], dtype=np.float64)
    camera = transform[:3, :3] @ np.asarray(rows, dtype=np.float64).T + transform[:3, 3:4]
    if (camera[2] <= 1e-3).any():
        return None
    uv = (intrinsics @ camera)[:2] / camera[2]
    return uv[0], uv[1], camera[2]


def from_sight_line(entry, u, v, depth) -> np.ndarray:
    """The inverse of sight_line(): (u, v, range) back to ego-frame centres."""
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(entry["camera"]["intrinsics"], dtype=np.float64)
    # atleast_1d: this is called both on a whole track and on one sample at a
    # time, and a 0-d scalar would not stack against the row of ones.
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    rays = np.linalg.solve(intrinsics, np.stack([u, v, np.ones(len(u))]))
    camera = rays / rays[2] * np.atleast_1d(np.asarray(depth, dtype=np.float64))
    return (transform[:3, :3].T @ (camera - transform[:3, 3:4])).T


def scale_inliers(values, window: int, inlier_frac: float) -> np.ndarray:
    """Which samples agree with the median of their own window.

    The companion to robust_scale_filter: the same test, reported instead of
    applied, so a caller can tell which frames the filter treated as outliers.
    """
    values = np.asarray(values, dtype=np.float64)
    half = max(window // 2, 0)
    out = np.empty(len(values), dtype=bool)
    for i in range(len(values)):
        median = float(np.median(values[max(0, i - half):i + half + 1]))
        out[i] = abs(values[i] - median) <= inlier_frac * abs(median)
    return out


def view_weights(track, min_visible=VOTE_MIN_VISIBLE) -> np.ndarray:
    """How much each of a track's frames is worth as evidence about the object.

    Detection score times projected area, restricted to the views that fit inside
    the image if the track has any. Both halves matter and both are argued for at
    length in majority_name, which is where this started: a vehicle 40 m away is a
    few dozen pixels with nothing readable on it, and a vehicle close enough to
    overflow the border is a slab of painted panel -- larger than every honest
    view of it and less informative than any of them.

    Area, not linear size, because how much a frame can tell you scales with how
    many pixels there were to look at. A box with no silhouette at all (one
    straddling the image plane) falls back to weight 1, which leaves it unable to
    decide anything against a box that has one.
    """
    scores = track["score"]
    visible = track.get("visible") or [None] * len(scores)
    sizes = track.get("size") or [None] * len(scores)

    def area_of(index):
        if visible[index] is not None:
            return visible[index][0]
        return float(sizes[index]) ** 2 if sizes[index] else 1.0

    untruncated = [i for i, v in enumerate(visible)
                   if v is not None and v[1] >= min_visible]
    weights = np.zeros(len(scores), dtype=np.float64)
    for i in (untruncated or range(len(scores))):
        weights[i] = float(scores[i]) * area_of(i)
    return weights


def majority_name(track, min_visible=VOTE_MIN_VISIBLE) -> str:
    """The track's label, as a vote weighted by detection score and apparent size.

    A single frame's relabelling should not redefine an object that forty other
    frames agree about, and the class decides which size prior the box was built
    from -- so a stray one changes its proportions, not just its caption.

    Every frame votes with its view_weights() weight rather than once, so the
    frames where the object is large, close and wholly in shot outvote the ones
    where it is a smudge near the horizon. Those are the frames whose label is
    worth having: a vehicle 40 m away is a few dozen pixels with no visible bed,
    cab or axle count, and the detector calls almost all of them "car" -- while
    the same vehicle at 12 m fills a third of the frame and is read correctly. An
    unweighted vote lets the twenty uninformative frames outnumber the five
    informative ones, which is how a truck labelled truck, truck, truck, bus over
    its four best frames ends up captioned "car" and rebuilt on the car prior.
    """
    names = track["names"]
    weights = {}
    for name, weight in zip(names, view_weights(track, min_visible)):
        weights[name] = weights.get(name, 0.0) + float(weight)
    return max(weights.items(), key=lambda kv: kv[1])[0]


def class_proportions(name):
    """(length, width, height) the lifter would have given `name`, or None."""
    length = CLASS_LENGTH_PRIOR.get(name)
    cross = CLASS_SIZE_PRIOR.get(name)
    return None if length is None or cross is None else np.array([length, *cross], float)


def normalise_proportions(dims, names, label):
    """Re-shapes every box in a track to the proportions of its settled class.

    _anchor_box_to_mask sizes a box by scaling its class prior's proportions onto
    the mask, so a single frame relabelled car -> truck changes the box's shape,
    not just its caption: height went from a third of length to nearly half, which
    is the box visibly flipping from lying down to standing up. Majority voting
    fixes the label but not the geometry already baked in from the wrong prior,
    and the extent median cannot help either -- a class flip is a step, and the
    median preserves steps on purpose so that collisions survive it.

    Overall size is held fixed (the geometric mean of the three extents is
    preserved) so only the shape changes; the absolute scale stays whatever the
    mask fit made it, and is left to the extent filter.
    """
    target = class_proportions(label)
    if target is None:
        return dims
    out = dims.copy()
    for i, name in enumerate(names):
        source = class_proportions(name)
        if source is None or name == label:
            continue
        reshaped = dims[i] * (target / source)
        scale = np.exp(np.mean(np.log(np.maximum(dims[i], 1e-6)))) / \
            np.exp(np.mean(np.log(np.maximum(reshaped, 1e-6))))
        out[i] = reshaped * scale
    return out


def robust_scale_filter(values, window: int, inlier_frac: float) -> np.ndarray:
    """Median-then-trimmed-mean filter along a track's extent.

    Same shape of filter as robust_angle_filter and for the same reason: extent
    errors are isolated blow-ups, not gentle wander (12% of frames change an
    extent by more than a quarter, and the 99th percentile is a doubling). A
    median removes those while still tracking the genuine, steady growth of an
    object being approached, and follows a real step -- a class settling, a
    collision -- within half a window.

    The inlier band is relative because extent scales with range: a fixed metric
    threshold would either clip an approaching object's real growth or wave
    through an outlier on a distant one.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    half = max(window // 2, 0)
    for i in range(len(values)):
        neighbourhood = values[max(0, i - half):i + half + 1]
        median = float(np.median(neighbourhood))
        inliers = neighbourhood[np.abs(neighbourhood - median) <= inlier_frac * abs(median)]
        out[i] = float(inliers.mean()) if len(inliers) else median
    return out


def smooth_track(rows, kernel, window: int, inlier_rad: float, max_yaw_rate: float):
    smoothed = rows.copy()
    smoothed[:, POS] = smooth_1d(rows[:, POS], kernel)
    smoothed[:, DIM] = smooth_1d(rows[:, DIM], kernel)
    # Heading gets the robust filter rather than the Gaussian: its errors are
    # isolated outliers, not the low-amplitude wander that centre and extent show.
    smoothed[:, YAW] = robust_angle_filter(rows[:, YAW], window, inlier_rad)
    # And then the rate limit, for the errors that are not isolated: a mode flip
    # holds for the rest of the track and outlives any window. Rows here are on a
    # contiguous grid (resample() filled the dropouts), so every step is one
    # frame. No weights either -- resampling has already mixed interpolated
    # samples in among the measured ones, so the segments are weighed by length.
    # The extent needs no equivalent: the Gaussian above is itself a rate limit.
    smoothed[:, YAW], _ = rate_limit_angle(smoothed[:, YAW], np.arange(len(rows)),
                                           max_yaw_rate)
    return smoothed


# A bridged link has to look like the object carrying on, not like the track
# stepping sideways onto its neighbour. The association gate is a *per-frame*
# allowance applied unchanged however many frames a link spans, so across a gap
# it selects whatever has barely moved -- and over half a second "barely moved"
# describes the next vehicle back in a stream of traffic, not the one that was
# there. Measured over the ambulance clip, the wider the gap the worse it gets:
# at a 1-frame gap the bridged pair moves 0.65 of what one object crossing it
# would, at 3 frames 0.44, at 5 frames 0.18 -- by which point 91% of bridges have
# the pair sitting still while the track's own object was crossing the frame.
#
# So a bridge is checked against the track's own velocity either side of it. The
# tolerance is generous (a full speed's worth, floored so a slow object is not
# judged on a tiny relative number) because it only has to catch the gross case:
# a track running left at 27 px/frame whose bridge goes right at 5. The
# refusals are heavily concentrated in the wide gaps, which is the point -- this
# lets --max_fill stay a duration and makes the gap width police itself.
BRIDGE_VELOCITY_TOL = 0.8        # of the track's own speed
BRIDGE_VELOCITY_FLOOR_PX = 8.0   # ... but never tighter than this, per frame
# Samples either side of the bridge that the velocity is taken over. Short,
# because it is the velocity *at* the gap that matters, not the track's average.
BRIDGE_VELOCITY_SPAN = 3


def _track_pixels(entries, samples):
    """Projected centre of each of a track's samples, or None where behind us."""
    out = []
    for entry, (_, row, _) in zip(entries, samples):
        ray = sight_line(entry, np.asarray(row, dtype=np.float64)[POS][None, :])
        out.append(None if ray is None else np.array([ray[0][0], ray[1][0]]))
    return out


def _bridge_follows_track(pixels, samples, k) -> bool:
    """Is the link from sample k to k+1 the object continuing on its way?

    Compares the bridge's own per-frame velocity against the median velocity of
    the adjacent-frame steps around it. Abstains -- returns True -- wherever the
    evidence is not there: a centre behind the camera, no adjacent steps to take
    a velocity from, or an object that is not moving across the image, which has
    no direction for a bridge to contradict.
    """
    if pixels[k] is None or pixels[k + 1] is None:
        return True
    steps = []
    for j in list(range(max(0, k - BRIDGE_VELOCITY_SPAN), k)) + \
            list(range(k + 1, min(len(pixels) - 1, k + 1 + BRIDGE_VELOCITY_SPAN))):
        if (samples[j + 1][0] - samples[j][0] == 1
                and pixels[j] is not None and pixels[j + 1] is not None):
            steps.append(pixels[j + 1] - pixels[j])
    if not steps:
        return True
    velocity = np.median(np.stack(steps), axis=0)
    speed = float(np.linalg.norm(velocity))
    gap = samples[k + 1][0] - samples[k][0]
    bridge = (pixels[k + 1] - pixels[k]) / gap
    tolerance = max(BRIDGE_VELOCITY_TOL * speed, BRIDGE_VELOCITY_FLOOR_PX)
    return float(np.linalg.norm(bridge - velocity)) <= tolerance


def _gap_fills(samples, name, max_fill, manual=None, entries=None):
    """Interpolated boxes for the frames in the middle of a track that carry none.

    The mirror of --min_track_len. A detector that fires once on nothing leaves a
    box that pops into existence; one that misses an object it has been tracking
    for a second leaves a hole, and a box vanishing for two frames and coming
    back reads exactly as badly. Both are the detector being unreliable over a
    handful of frames, and both are answered from the same fact -- how long the
    track around them is.

    This is the one place the in-place path moves a centre, and it has to: there
    is no mask in a frame with no detection, so nothing to anchor to. Every
    filled frame is bracketed by two real observations at most --max_fill apart,
    which at 20 fps is 0.15 s, so a linear interpolation between them is well
    inside the error of the boxes it is bridging.

    :param samples: (frame index, box row, score) per observed frame, in order
    :param manual: the track's annotator id, or None -- stamped on every box
        filled here so a drawn track stays one track when this is re-run
    :param entries: the frames' entries, aligned with `samples`, so a bridge can be
        checked against the track's own motion. None skips that check.
    :returns: ((frame index, box row, name, score, manual id) per filled frame,
        number of frames left unfilled because the bridge failed that check)
    """
    if max_fill <= 0:
        return [], 0
    pixels = _track_pixels(entries, samples) if entries is not None else None
    fills, refused = [], 0
    for k, ((frame_a, row_a, score_a), (frame_b, row_b, score_b)) in enumerate(
            zip(samples, samples[1:])):
        missing = frame_b - frame_a - 1
        if not 1 <= missing <= max_fill:
            continue
        if pixels is not None and not _bridge_follows_track(pixels, samples, k):
            refused += missing
            continue
        start, end = np.asarray(row_a, float), np.asarray(row_b, float)
        for step in range(1, missing + 1):
            weight = step / (missing + 1.0)
            row = (1.0 - weight) * start + weight * end
            # Heading on the doubled angle, as everywhere else here: yaw is only
            # defined up to +-pi, so lerping it raw would swing a box through a
            # half-turn whenever the two ends picked different representatives.
            row[YAW] = np.angle((1.0 - weight) * np.exp(2j * start[YAW])
                                + weight * np.exp(2j * end[YAW])) / 2.0
            fills.append((frame_a + step, [float(v) for v in row], name,
                          float((1.0 - weight) * score_a + weight * score_b),
                          manual))
    return fills, refused


# How much of an interpolated box may be swallowed by a real one in the same
# frame before it is discarded as a duplicate. The same containment test
# lift_frames_to_3d._suppress_duplicate_masks applies to overlapping masks, for
# the same reason and at a looser threshold: real detections overlap all the
# time (a car behind a car), so 0.7 would delete real boxes -- but only fills are
# tested here, and a fabricated box sitting almost wholly inside a measured one
# is a duplicate whatever the two tracks think.
FILL_CONTAINMENT = 0.7


def _drop_shadowed_fills(out, frames, fills):
    """Discards interpolated boxes that land on an object already drawn.

    A fill exists to cover a frame where the detector saw nothing. When one lands
    on top of a box that *is* in that frame, the premise is gone: the detector
    did see the object, under another track, and inserting a second box on it
    draws the same car twice at two sizes.

    This is what frame 82 of the ambulance clip showed. The white SUV crossing
    ahead was detected at 0.82 and its own track deleted by --min_track_len,
    while two other tracks -- each of which had linked two different cars across
    a gap -- interpolated across that frame and put their invented boxes on it.
    The result was one car wearing two boxes, both too small, and its own box
    absent. --min_track_len no longer deletes it, so all three now coincide; this
    removes the two that were never measured.

    Tested against the frame's surviving *measured* boxes only, not against other
    fills: two fills that agree with each other are one track's dropout seen
    twice, which is a different fault and not one to arbitrate by deletion.

    :returns: (fills to keep, number discarded)
    """
    by_frame = {}
    for fill in fills:
        by_frame.setdefault(fill[0], []).append(fill)

    kept = []
    for fi, frame_fills in by_frame.items():
        entry = out[frames[fi]]
        drawn = [b for b in projected_bounds(entry) if b is not None]
        for fill in frame_fills:
            bounds = projected_bounds(entry, [fill[1]])[0]
            if bounds is None or not _is_contained(bounds, drawn):
                kept.append(fill)
    return kept, len(fills) - len(kept)


def _is_contained(bounds, others) -> bool:
    """True if `bounds` sits inside one of `others` by more than FILL_CONTAINMENT.

    Directional: it asks whether the *fill* is redundant, not whether the two
    overlap. Measuring containment against the smaller of the pair -- which is
    what this did at first -- makes it symmetric, and then a fill is thrown away
    for covering a box far smaller than itself. That is the wrong way round, and
    on the ambulance clip it deleted the ambulance twice: at frames 44 and 47 the
    detector returned only the lower half of it, as a 41 px "car" rather than the
    120 px truck of every frame either side, and the fill that would have carried
    the real box across those two frames was discarded for landing on that
    fragment. The vehicle shrank to a third of itself for one frame, twice.

    So the fill has to be the contained one. A fill swallowed by a box of its own
    size or larger is a duplicate and goes; a fill that swallows something much
    smaller is a whole object drawn over a piece of one, and stays.
    """
    low, high = bounds
    area = float(np.prod(np.maximum(high - low, 1e-6)))
    for other_low, other_high in others:
        overlap = np.prod(np.maximum(np.minimum(high, other_high)
                                     - np.maximum(low, other_low), 0.0))
        if not overlap:
            continue
        if overlap / area > FILL_CONTAINMENT:
            return True
    return False


def _insert_fills(out, frames, fills) -> int:
    """Appends the interpolated boxes to their frames' parallel lists.

    Appended rather than inserted in track order: nothing downstream reads any
    meaning into a box's position within a frame, and appending leaves every
    slot index the filtering pass used still valid.
    """
    for fi, row, name, score, manual in fills:
        entry = out[frames[fi]]
        entry.setdefault("boxes", []).append(row)
        entry.setdefault("names", []).append(name)
        if "scores" in entry:
            entry["scores"].append(score)
        if "manual_tracks" in entry:
            entry["manual_tracks"].append(manual)
    return len(fills)


def _drop_slots(out, frames, dropped) -> int:
    """Removes the listed box positions from each frame's parallel lists.

    A box detected in one or two frames of a clip is a flicker, not an object:
    the detector fires once on an oncoming vehicle across the divider, or on a
    reflection, and the box pops into existence for a frame and is gone -- which
    reads far worse in a video than the missing detection would. Association
    already knows how long each object was actually seen, so this is the natural
    place to act on it: a track that never lasted is removed everywhere it
    appeared, rather than faded or interpolated into something it never was.

    Not folded into the loop above: `boxes`, `names` and `scores` are addressed
    by index throughout that pass, so deletions have to wait until the last
    write is done. Descending order for the same reason -- popping slot 3 before
    slot 5 would move slot 5.
    """
    removed = 0
    for fi, slots in dropped.items():
        entry = out[frames[fi]]
        for slot in sorted(slots, reverse=True):
            for field in ("boxes", "names", "scores", "manual_tracks"):
                if field in entry and slot < len(entry[field]):
                    entry[field].pop(slot)
            removed += 1
    return removed


def _light_pixels(entry):
    """Per traffic light: (u, v, apparent size in px), or None if behind us.

    The apparent size is recovered from the range rather than measured, because
    the mask does not survive into boxes_3d.json -- but the range was fitted
    through the very same fixed height in _traffic_light_position, so
    fy * H / range gives back exactly the mask extent it came from.
    """
    transform = np.asarray(entry["camera"]["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(entry["camera"]["intrinsics"], dtype=np.float64)
    out = []
    for light in entry.get("traffic_lights", []):
        position = np.asarray(light["position"], dtype=np.float64)
        camera = transform[:3, :3] @ position + transform[:3, 3]
        if camera[2] <= 1e-3:
            out.append(None)
            continue
        uv = (intrinsics @ camera)[:2] / camera[2]
        size = float(intrinsics[1, 1] * TRAFFIC_LIGHT_DIMS[2] / camera[2])
        out.append((float(uv[0]), float(uv[1]), size))
    return out


def _associate_lights(out, frames, max_gap):
    """Greedy nearest-centroid tracks over the frames' traffic lights.

    A cut-down `associate`: lights carry no extent to gate on and no class to
    match across, so this is the distance test alone, and the gate is the one
    TRAFFIC_LIGHT_GATE_FRAC describes.
    """
    tracks, active = [], []
    for fi, frame in enumerate(frames):
        entry = out[frame]
        projected = _light_pixels(entry)
        pairs = []
        for li, point in enumerate(projected):
            if point is None:
                continue
            for ti in active:
                track = tracks[ti]
                distance = float(np.hypot(point[0] - track["pixel"][0],
                                          point[1] - track["pixel"][1]))
                gate = min(max(TRAFFIC_LIGHT_GATE_FRAC * max(point[2], track["size"]),
                               TRAFFIC_LIGHT_MIN_GATE_PX), TRAFFIC_LIGHT_MAX_GATE_PX)
                if distance <= gate:
                    pairs.append((distance / gate, li, ti))
        pairs.sort(key=lambda pair: pair[0])

        used_l, used_t, assigned = set(), set(), {}
        for _, li, ti in pairs:
            if li in used_l or ti in used_t:
                continue
            used_l.add(li); used_t.add(ti); assigned[li] = ti

        for li, point in enumerate(projected):
            if point is None:
                continue
            if li in assigned:
                track = tracks[assigned[li]]
            else:
                track = {"idx": [], "slot": [], "pixel": None, "size": 0.0}
                tracks.append(track)
            track["idx"].append(fi)
            track["slot"].append(li)
            track["pixel"] = point[:2]
            track["size"] = point[2]

        active = [ti for ti, track in enumerate(tracks)
                  if fi - track["idx"][-1] <= max_gap]
    return tracks


def _vote_state(lights):
    """The state a track settles on: its members' states weighted by score.

    "unknown" never wins a vote -- it is an abstention, not a colour -- so a
    track is only unknown when every one of its frames abstained. Ties go to
    whichever colour the more confident detections carried, which is what the
    score weighting is for; on a light seen through a change it means the state
    is the one held for the larger part of the track, and neither that nor a
    per-frame reading is right about the moment it changed.
    """
    weights = {}
    for light in lights:
        state = light.get("state", "unknown")
        if state == "unknown":
            continue
        weights[state] = weights.get(state, 0.0) + float(light.get("score", 1.0))
    if not weights:
        return "unknown"
    return max(weights, key=weights.get)


def filter_traffic_lights(out, frames, min_track_len, max_gap, max_fill):
    """Tracks the traffic lights, then drops flickers, fills holes and votes.

    Mutates `out` in place and returns (dropped, filled, revoted). The three
    passes are ordered as they are for boxes: slots are addressed by position,
    so deletions wait until every other write is done, and fills are appended
    afterwards where they cannot renumber anything.
    """
    # dict(data[frame]) upstream is shallow, so these lists and the dicts in
    # them are still the caller's until they are copied here.
    for frame in frames:
        out[frame]["traffic_lights"] = [dict(light)
                                        for light in out[frame].get("traffic_lights", [])]

    tracks = _associate_lights(out, frames, max_gap)
    dropped, fills, revoted = {}, [], 0
    for track in tracks:
        lights = [out[frames[fi]]["traffic_lights"][slot]
                  for fi, slot in zip(track["idx"], track["slot"])]
        if min_track_len and len(track["idx"]) < min_track_len:
            for fi, slot in zip(track["idx"], track["slot"]):
                dropped.setdefault(fi, set()).add(slot)
            continue

        state = _vote_state(lights)
        for light in lights:
            revoted += light.get("state", "unknown") != state
            light["state"] = state

        if max_fill <= 0:
            continue
        for (fi_a, light_a), (fi_b, light_b) in zip(zip(track["idx"], lights),
                                                    zip(track["idx"][1:], lights[1:])):
            missing = fi_b - fi_a - 1
            if not 1 <= missing <= max_fill:
                continue
            start = np.asarray(light_a["position"], dtype=np.float64)
            end = np.asarray(light_b["position"], dtype=np.float64)
            for step in range(1, missing + 1):
                weight = step / (missing + 1.0)
                fills.append((fi_a + step, {
                    "position": [float(v) for v in (1.0 - weight) * start + weight * end],
                    "state": state,
                    "score": float((1.0 - weight) * light_a.get("score", 1.0)
                                   + weight * light_b.get("score", 1.0)),
                }))

    removed = 0
    for fi, slots in dropped.items():
        lights = out[frames[fi]]["traffic_lights"]
        for slot in sorted(slots, reverse=True):
            lights.pop(slot)
            removed += 1
    for fi, light in fills:
        out[frames[fi]]["traffic_lights"].append(light)
    return removed, len(fills), revoted


def _report_dropped(dropped_tracks, limit=10):
    """Name what --min_track_len deleted instead of only counting it.

    A count is invisible. The same line deletes a one-frame flicker on a
    reflection and a correct detection of a real vehicle the detector only
    caught a glimpse of, and the second is the one worth going and drawing a
    keyframe over -- but "1 box dropped" does not say which frame to open.
    """
    for name, frame_list, score in sorted(dropped_tracks)[:limit]:
        where = (f"{frame_list[0]}..{frame_list[-1]}" if len(frame_list) > 3
                 else " ".join(frame_list))
        print(f"    {name:<14} {where}  (best score {score:.2f})")
    if len(dropped_tracks) > limit:
        print(f"    ... and {len(dropped_tracks) - limit} more")
    if dropped_tracks:
        print("    -- to keep one, draw a keyframe near it in annotate_boxes.py: a drawn\n"
              "       identity is exempt from --min_track_len, and apply_manual_boxes\n"
              "       adopts the detections around it into the same track.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Run directory holding boxes_3d.json.")
    ap.add_argument("--out", type=str, default=None,
                    help="Where to write. Default: <output_dir>/boxes_3d_smoothed.json. "
                         "Pass <run>/boxes_3d.json to render it with raster_frames.py.")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian width in frames. 0 disables smoothing.")
    ap.add_argument("--max_dist", type=float, default=3.0,
                    help="Association gate: max centre movement between frames, metres.")
    ap.add_argument("--max_px", type=float, default=400.0,
                    help="Ceiling on the association gate in pixels, used by --in_place "
                         "instead of --max_dist. The gate itself is --gate_extent_frac of "
                         "the object's projected size; this only caps it for the largest "
                         "objects in frame. 0 falls back to metres.")
    ap.add_argument("--gate_extent_frac", type=float, default=GATE_EXTENT_FRAC,
                    help="Association gate as a fraction of the object's own projected "
                         "size, so a near vehicle is allowed to move further between "
                         "frames than a distant one. Clamped to [--min_px, --max_px].")
    ap.add_argument("--min_px", type=float, default=MIN_GATE_PX,
                    help="Floor under the scaled gate, in pixels, so a distant object is "
                         "not gated tighter than the jitter of its own box fit.")
    ap.add_argument("--max_size_ratio", type=float, default=MAX_SIZE_RATIO,
                    help="Largest ratio between two boxes' projected sizes still taken to "
                         "be one object. This is what stops a near truck being adopted by "
                         "a track of distant cars its centre happens to land on.")
    ap.add_argument("--vote_min_visible", type=float, default=VOTE_MIN_VISIBLE,
                    help="Fraction of its silhouette a box must have inside the image "
                         "before its class label joins the track's vote, so a vehicle too "
                         "close to fit in frame does not name itself. A track with no such "
                         "view falls back to voting over its least-clipped frames.")
    ap.add_argument("--size_weight", type=float, default=SIZE_COST_WEIGHT,
                    help="Weight on the size-mismatch term in the association cost, in "
                         "octaves of size ratio per gate width of distance. 0 keeps the "
                         "hard --max_size_ratio test but ranks purely on distance.")
    ap.add_argument("--max_gap", type=int, default=2,
                    help="Frames a track may go undetected and still continue.")
    ap.add_argument("--min_len", type=int, default=2,
                    help="Drop tracks seen in fewer than this many frames.")
    ap.add_argument("--min_track_len", type=int, default=3,
                    help="--in_place only: drop every box of a track detected in fewer than "
                         "this many frames. 0 keeps them all.")
    ap.add_argument("--max_fill", type=int, default=3,
                    help="--in_place only: interpolate a track's box across dropouts of up "
                         "to this many consecutive frames, so an object the detector loses "
                         "for a moment does not blink out. 0 leaves the holes.")
    ap.add_argument("--no_fill_gaps", action="store_true",
                    help="Keep detection dropouts instead of interpolating across them.")
    ap.add_argument("--in_place", "--yaw_only", dest="in_place", action="store_true",
                    help="Filter heading, extent and range, writing every input box back "
                         "where it still projects to the same place. Use this downstream of "
                         "lift_frames_to_3d.py's mask anchoring: the anchoring fixes where a "
                         "box sits in the image, and everything filtered here -- heading, "
                         "extent, and how far down its own sight line the box lies -- is a "
                         "field it leaves free to jitter. The one exception is "
                         "--dampen_centre_px, which may move a box across its sight line by "
                         "at most that many pixels.")
    ap.add_argument("--no_size", action="store_true",
                    help="Filter heading only, leaving extent exactly as lifted.")
    ap.add_argument("--size_window", type=int, default=5,
                    help="Frames in the extent median window (odd).")
    ap.add_argument("--size_inlier_frac", type=float, default=0.25,
                    help="Extents within this fraction of the window median are averaged; "
                         "the rest are discarded as outliers.")
    ap.add_argument("--max_shrink", type=float, default=0.90,
                    help="Floor on how far the extent filter may shrink a box, as a "
                         "fraction of its lifted size. Asymmetric on purpose: "
                         "lift_frames_to_3d.py already shrank each box to the edge of its "
                         "mask-coverage floor, so shrinking further is the only direction "
                         "that breaks coverage, while growing never does. Applied only on "
                         "frames whose lifted size agrees with its own window, since a "
                         "frame whose mask blew up must not be allowed to set its own "
                         "floor. Set to 0 to let the filter shrink freely.")
    ap.add_argument("--yaw_window", type=int, default=5,
                    help="Frames in the heading median window (odd). Larger rejects longer "
                         "bursts of flicker but delays a genuine turn by half the window.")
    ap.add_argument("--max_yaw_rate", type=float, default=MAX_YAW_RATE_DEG,
                    help="Largest heading change a box may make between two frames, in "
                         "degrees. A step past this is not slowed down but dropped, and "
                         "the track carries on from the heading it already held -- see "
                         "MAX_YAW_RATE_DEG for why a slew limit is not enough. 0 disables "
                         "the limit. Scale it with the frame rate: the default is set for "
                         "the 10 Hz this pipeline processes at.")
    ap.add_argument("--max_size_rate", type=float, default=MAX_SIZE_RATE,
                    help="Largest change in a box's angular size between two frames, as a "
                         "fraction. Clamped rather than dropped, so a box that really is "
                         "growing catches up within a frame or two. 0 disables the limit. "
                         "Scale it with the frame rate, as with --max_yaw_rate.")
    ap.add_argument("--dampen_range", type=float, default=DAMPEN_RANGE,
                    help="Deadband on how far away a box is, as a fraction of its range. "
                         "Its range is the one quantity nothing measures -- the mask fixes "
                         "where a box sits in the image and says nothing about depth -- and "
                         "it is the one that behaves like noise (46%% of consecutive steps "
                         "reverse). The box is moved along its own sight line, so it still "
                         "projects where it did. 0 disables.")
    ap.add_argument("--dampen_size", type=float, default=DAMPEN_SIZE,
                    help="Deadband on a box's angular size, as a fraction. Removes the "
                         "back-and-forth the rate limit is too coarse to see. 0 disables.")
    ap.add_argument("--dampen_extent", type=float, default=DAMPEN_EXTENT,
                    help="Deadband on a box's metric length/width/height, as a fraction. "
                         "A vehicle is rigid, so this holds l/w/h still until the "
                         "measurement leaves the band. 0 disables.")
    ap.add_argument("--range_inlier_frac", type=float, default=RANGE_INLIER_FRAC,
                    help="Ranges within this fraction of their --size_window median are "
                         "averaged; the rest are replaced by it. Removes the monocular "
                         "excursions that a deadband cannot, and which show up as a box "
                         "changing size rather than moving. 0 disables.")
    ap.add_argument("--dampen_centre_px", type=float, default=DAMPEN_CENTRE_PX,
                    help="Deadband on where a box's centre sits in the image, in pixels. "
                         "The only damping that moves a box off the mask it was anchored "
                         "to, and it can move it no further than this. 0 disables.")
    ap.add_argument("--yaw_inlier_deg", type=float, default=15.0,
                    help="Headings within this of the window median are averaged; the rest "
                         "are discarded as outliers.")
    args = ap.parse_args()
    inlier_rad = np.radians(args.yaw_inlier_deg) * 2.0   # doubled-angle domain

    output_dir = Path(args.output_dir)
    src = output_dir / "boxes_3d.json"
    dst = Path(args.out) if args.out else output_dir / "boxes_3d_smoothed.json"
    data = json.loads(src.read_text())
    frames = sorted(data)

    # A dropout can only be filled if the track survived it in the first place,
    # so association has to look at least as far across a gap as --max_fill does.
    max_gap = max(args.max_gap, args.max_fill) if args.in_place else args.max_gap
    tracks = associate(frames, data, args.max_dist, max_gap,
                       cross_class=args.in_place,
                       max_px=args.max_px if args.in_place else None,
                       gate_extent_frac=args.gate_extent_frac,
                       min_px=args.min_px,
                       max_size_ratio=args.max_size_ratio,
                       size_weight=args.size_weight)

    if args.in_place:
        out = {frame: dict(data[frame]) for frame in frames}
        for frame in frames:
            out[frame]["boxes"] = [list(box_schema.from_any(b))
                                   for b in data[frame].get("boxes", [])]
            out[frame]["names"] = list(data[frame].get("names", []))
            # Copied like the other two: dict() above is shallow, and _drop_slots
            # pops from these lists.
            if "scores" in out[frame]:
                out[frame]["scores"] = list(data[frame]["scores"])
            # Parallel to boxes like names and scores. Normalised to length here
            # so the drop and fill passes can keep it in step without having to
            # special-case a boxes_3d.json lifted before the field existed.
            ids = list(data[frame].get("manual_tracks", []))
            out[frame]["manual_tracks"] = \
                ids + [None] * (len(out[frame]["boxes"]) - len(ids))
        moved, resized, relabelled = [], [], 0
        # Steps the two rate limits refused, counted for the report: they are the
        # events the user sees as a box snapping round or popping in size.
        flips, pops, nudged = 0, 0, []
        # Slots to delete, per frame index. Collected while filtering and applied
        # afterwards, because every other write here addresses a box by its
        # position in the frame's list -- removing one mid-pass would renumber
        # the ones behind it.
        dropped, dropped_tracks = {}, []
        fills = []
        # Frames a dropout left unfilled because the boxes bracketing it did
        # not move like one object -- see BRIDGE_VELOCITY_TOL.
        strayed = 0
        for t in tracks:
            # A drawn box is exempt: --min_track_len exists to delete a
            # detector firing once on nothing, and someone drawing a box is the
            # opposite of that -- one frame of it is still a deliberate one.
            if (args.min_track_len and len(t["idx"]) < args.min_track_len
                    and t["manual"] is None):
                for fi, slot in zip(t["idx"], t["slot"]):
                    dropped.setdefault(fi, set()).add(slot)
                dropped_tracks.append((t["name"], [frames[i] for i in t["idx"]],
                                       max(t["score"])))
                continue
            if len(t["idx"]) < 2:
                continue
            # Split each centre into the part the mask anchored and the part it
            # did not, and damp only the second. Moving a box along its own sight
            # line leaves it projecting exactly where it was, so the anchoring
            # survives untouched -- and the extent is rescaled with the range
            # below to hold the angular size, which is the pairing the lifter
            # made for the same reason. The across-ray damping is the one place
            # this does touch the anchor, which is why its band is in pixels and
            # small: the box can end up at most --dampen_centre_px off its mask,
            # and the mask's own centre wanders further than that between frames.
            entries = [out[frames[fi]] for fi in t["idx"]]
            rays = [sight_line(e, np.asarray(b[POS], dtype=np.float64)[None, :])
                    for e, b in zip(entries, t["box"])]
            range_scale, centres = None, None
            if all(r is not None for r in rays):
                u = np.array([r[0][0] for r in rays])
                v = np.array([r[1][0] for r in rays])
                ranges = np.array([r[2][0] for r in rays])
                # Median-then-trimmed-mean first, deadband second: the deadband
                # bounds its output to within a band of its input, so an
                # excursion that leaves the band and stays out survives it
                # intact. See RANGE_INLIER_FRAC.
                filtered_range = (
                    robust_scale_filter(ranges, args.size_window,
                                        args.range_inlier_frac)
                    if args.range_inlier_frac > 0 else ranges)
                damped = dampen_ratio(filtered_range, args.dampen_range)
                centres = np.stack([
                    from_sight_line(e, uu, vv, rr)[0] for e, uu, vv, rr in
                    zip(entries, dampen(u, args.dampen_centre_px),
                        dampen(v, args.dampen_centre_px), damped)])
                range_scale = (damped / ranges)[:, None]

            weights = view_weights(t, args.vote_min_visible)
            yaws = np.array([b[YAW] for b in t["box"]])
            filtered = robust_angle_filter(yaws, args.yaw_window, inlier_rad)
            # The median removes the flicker that comes back; the rate limit
            # removes the flip that does not. Median first, so the limiter reads
            # its steps off a heading that has already had the search's 5 deg
            # quantisation staircase averaged out of it.
            filtered, rejected = rate_limit_angle(
                filtered, t["idx"], np.radians(args.max_yaw_rate), weights)
            flips += rejected
            label = majority_name(t, args.vote_min_visible)
            dims = np.stack([b[DIM] for b in t["box"]])
            if not args.no_size:
                dims = normalise_proportions(dims, t["names"], label)
                lifted = dims
                # Filter the *angular* size, not the metric one. What pulses on
                # screen is dims/depth; a box's metric extent is that times a
                # monocular depth that jumps several metres between frames, and
                # the lifter scales the two together so the projection stays put.
                # Smoothing metric extent alone therefore breaks that pairing and
                # leaves the visible size just as jumpy (measured: projected-area
                # change per frame went 9% -> 8%, i.e. nothing).
                depth = np.maximum(np.asarray(t["depth"], dtype=np.float64), 1e-3)[:, None]
                angular = np.stack([robust_scale_filter(v, args.size_window,
                                                        args.size_inlier_frac)
                                    for v in (dims / depth).T], axis=1)
                dims = angular * depth
                # The floor only applies where the frame's own lifted size is an
                # inlier of its window. It is there because the lifter had
                # already shrunk each box to the edge of its mask-coverage floor
                # -- but that argument is about a box fitted to a mask that was
                # right, and on the frame where the mask spilled onto a parked
                # neighbour it lets the blow-up set its own floor and holds the
                # filter off exactly the frame the filter exists for. Measured on
                # wrongway/4: 21 of the 26 worst extent spikes were reinstated
                # this way after the median had already removed them.
                trusted = scale_inliers(lifted[:, 0] / depth[:, 0],
                                        args.size_window, args.size_inlier_frac)
                dims = np.where(trusted[:, None],
                                np.maximum(dims, lifted * args.max_shrink), dims)
                # Last word on extent, after the floor rather than before it, so
                # that nothing downstream can put a pop back. A box may still be
                # wrong here; it may not change by more than a real one could.
                angular, clamped = rate_limit_scale(dims / depth, t["idx"],
                                                    args.max_size_rate)
                pops += clamped
                # ... and the last word on how much of what is left is real. The
                # rate limit has already taken out the steps too big to be a
                # vehicle; this takes out the ones too small and too undecided to
                # be one, which after it is most of what is left (measured: 4% of
                # consecutive size steps still reverse direction, against 46% of
                # range steps).
                angular = np.stack([dampen_ratio(c, args.dampen_size)
                                    for c in angular.T], axis=1)
                dims = angular * depth
            if range_scale is not None:
                # Hold the angular size across the range change, in both branches
                # -- with --no_size these are the lifted extents, and they were
                # paired with the lifted range just the same.
                dims = dims * range_scale
            if not args.no_size and args.dampen_extent > 0:
                # A vehicle is rigid, so its l/w/h is one number per track and
                # not one per frame. This is the last thing to touch the extent,
                # after the range rescale, so nothing downstream can put the
                # drift back. See DAMPEN_EXTENT.
                dims = np.stack([dampen_ratio(c, args.dampen_extent)
                                 for c in dims.T], axis=1)
            samples = []
            for k, (fi, slot) in enumerate(zip(t["idx"], t["slot"])):
                row = out[frames[fi]]["boxes"][slot]
                moved.append(abs(np.angle(np.exp(1j * 2 * (filtered[k] - row[YAW]))) / 2))
                resized.append(np.max(np.abs(dims[k] - row[DIM]) / np.maximum(row[DIM], 1e-6)))
                if centres is not None:
                    nudged.append(float(np.linalg.norm(centres[k] - row[POS])))
                    row[POS] = [float(x) for x in centres[k]]
                row[YAW] = float(filtered[k])
                row[DIM] = [float(v) for v in dims[k]]
                if slot < len(out[frames[fi]]["names"]):
                    relabelled += out[frames[fi]]["names"][slot] != label
                    out[frames[fi]]["names"][slot] = label
                scores = out[frames[fi]].get("scores", [])
                samples.append((fi, list(row),
                                float(scores[slot]) if slot < len(scores) else 1.0))
            # Built from the rows as just written, so an interpolated box lands
            # between its neighbours' filtered heading and extent, not between
            # the raw fits they replaced.
            new_fills, refused = _gap_fills(samples, label, args.max_fill,
                                            t["manual"], entries)
            fills.extend(new_fills)
            strayed += refused

        removed = _drop_slots(out, frames, dropped)
        # After the drop, so the boxes a fill is checked against are the ones
        # that will actually be drawn, and before the insert, so a fill is never
        # tested against another fill.
        fills, shadowed = _drop_shadowed_fills(out, frames, fills)
        inserted = _insert_fills(out, frames, fills)
        # Same three passes over the frames' traffic lights, which have their own
        # association because they carry neither an extent nor a class to match on.
        #
        # Lights are filled across the whole association window rather than
        # --max_fill. That limit is small because a filled *box* invents motion
        # the detector never saw; a traffic light is bolted to a mast and has no
        # motion to invent, so interpolating its position between two sightings
        # only re-derives where a static thing already was. Holding it to
        # --max_fill leaves exactly the blink the pass exists to remove: on the
        # ambulance clip the light seen in frame 40 and again from 47 is one
        # track, and a 5-frame allowance cannot close a 6-frame hole.
        lights_dropped, lights_filled, lights_revoted = filter_traffic_lights(
            out, frames, args.min_track_len, max_gap, max_gap)
        _write(out, dst)
        moved = np.degrees(moved) if moved else np.zeros(1)
        resized = np.array(resized) if len(resized) else np.zeros(1)
        print(f"frames {len(frames)}  tracks {len(tracks)}  in-place filter "
              f"(yaw window={args.yaw_window}/{args.yaw_inlier_deg} deg, "
              f"size window={args.size_window}/{args.size_inlier_frac:.0%})")
        print(f"headings changed: median {np.median(moved):.1f} deg, "
              f"p90 {np.percentile(moved, 90):.1f} deg, max {moved.max():.1f} deg")
        print(f"extents changed:  median {np.median(resized):.0%}, "
              f"p90 {np.percentile(resized, 90):.0%}, max {resized.max():.0%}")
        print(f"impossible steps removed: {flips} heading flips over "
              f"{args.max_yaw_rate:.0f} deg/frame, {pops} size pops over "
              f"{args.max_size_rate:.0%}/frame")
        nudged = np.array(nudged) if len(nudged) else np.zeros(1)
        print(f"jitter damped out (range {args.dampen_range:.0%}, size "
              f"{args.dampen_size:.0%}, centre {args.dampen_centre_px:.0f} px): "
              f"centres moved median {np.median(nudged):.2f} m, "
              f"p90 {np.percentile(nudged, 90):.2f} m, max {nudged.max():.2f} m")
        print(f"boxes relabelled to their track's majority class: {relabelled}")
        print(f"boxes dropped with tracks shorter than {args.min_track_len} frames: "
              f"{removed} (in {len(dropped)} frames)")
        _report_dropped(dropped_tracks)
        print(f"boxes interpolated across dropouts of up to {args.max_fill} frames: "
              f"{inserted} ({shadowed} discarded as landing on a box already drawn, "
              f"{strayed} not filled because the dropout's two ends did not move "
              f"like one object)")
        print(f"traffic lights: dropped {lights_dropped} in tracks shorter than "
              f"{args.min_track_len} frames, interpolated {lights_filled} across "
              f"dropouts, restated {lights_revoted} to their track's voted state")
        print(f"wrote {dst}")
        return

    kept = [t for t in tracks
            if len(t["idx"]) >= args.min_len or t["manual"] is not None]

    kernel = gaussian_kernel(args.sigma) if args.sigma > 0 else None
    per_frame = {fi: {"boxes": [], "names": [], "scores": []} for fi in range(len(frames))}
    filled = 0
    for t in kept:
        grid, rows, scores, observed = resample(t, not args.no_fill_gaps)
        rows = (smooth_track(rows, kernel, args.yaw_window, inlier_rad,
                             np.radians(args.max_yaw_rate))
                if kernel is not None else rows)
        filled += int((~observed).sum())
        for fi, row, sc in zip(grid, rows, scores):
            per_frame[int(fi)]["boxes"].append([float(v) for v in row])
            per_frame[int(fi)]["names"].append(t["name"])
            per_frame[int(fi)]["scores"].append(float(sc))

    out = {}
    for fi, frame in enumerate(frames):
        entry = dict(data[frame])          # keeps camera, lanes, traffic_lights
        entry["boxes"] = per_frame[fi]["boxes"]
        entry["names"] = per_frame[fi]["names"]
        entry["scores"] = per_frame[fi]["scores"]
        out[frame] = entry
    _write(out, dst)

    before = sum(len(data[f].get("boxes", [])) for f in frames)
    after = sum(len(out[f]["boxes"]) for f in frames)
    print(f"frames {len(frames)}  tracks {len(tracks)} -> kept {len(kept)} "
          f"(dropped {len(tracks) - len(kept)} shorter than {args.min_len})")
    print(f"boxes {before} -> {after}  (+{filled} interpolated across dropouts)")
    print(f"sigma={args.sigma} frames, gate={args.max_dist} m, max_gap={args.max_gap}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
