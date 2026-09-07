#!/usr/bin/env python3
"""Drop the boxes stage 3 fits to the road and the ego's own bodywork.

    python drop_ego_artifacts.py --output_dir ../data/test/bike_crosswalk/1
    python drop_ego_artifacts.py --output_dir <run> --dry-run

Two failure modes put a box in the ego's face, and neither is a detection of
anything. They are worth removing before the navsim export, where a 5.8 m "car"
sitting 8 m ahead is an obstacle the planner has to avoid.

THE SLAB  (rule off by default -- see below)

The lifter fits a box to whatever mask it is handed, and when segmentation
bleeds off a vehicle onto the tarmac it is handed the road. What comes back is
one enormous box covering the foreground -- on bike_crosswalk frames 12-14, a
"car" 5.77 m long where that clip's cars are 0.75 m, filling half the frame and
rendered right under the ego.

Size alone cannot catch it, because "large and close" is also what a truck you
are following looks like. What catches it is size *relative to its own class in
its own clip*. Every clip has its own metric scale (UniDepth's focal length is
guessed per clip), so absolute metres mean nothing across clips -- but within
one clip a car seven times longer than the median car is not a car. Measured:

    bike_crosswalk  the slab   car   5.77 / 0.75 median = 7.6x   dropped
    too_close       the truck  truck 7.34 / 4.35 median = 1.7x   kept
    too_close       largest    truck 11.52 / 4.35       = 2.6x   kept
    bike_crosswalk  largest    bus   7.69 / 5.28        = 1.5x   kept

That gap looked like room for a threshold, and it is not. On missing_blinker the
same rule at 4.0 deleted 114 boxes that are all real cars -- visibly, a Prius
directly ahead and traffic crossing an intersection. The reason is that a clip's
median conflates near and far: that clip's cars measure 0.82 at 10-20 m and 3.99
at 20-40 m, so the median (0.89) describes the distant ones and every near car
looks like an outlier against it. Conditioning the reference on range helps and
does not save it -- a real car at 15.7 m is 6.7x its own range band's median,
against the bike_crosswalk slab's 7.4x. Those two cannot be separated by any
threshold.

So --max_class_ratio defaults to 0, which turns this rule OFF, and the three
bike_crosswalk slabs are not caught by anything here. That is the right trade:
losing 3 artifacts on one clip costs less than deleting 114 real vehicles on
another, and the two remaining rules are geometric rather than statistical --
they ask where a box is, not whether it is unusual. Turn it on per clip
(--max_class_ratio 4) only after looking at what it takes.

THE ENGULFING BOX

The third is a box that swallows the whole frame. On yield_runway the detector
reads the ego's own red bonnet as a car, and the lift puts it 8.9 m ahead at
9.2 m long: its projection is [-54, -866, 1973, 1089] on a 1920x1080 frame, so
every pixel of the image is inside it and the whole overlay tints with it. It
survives the size rule -- 2.88x the clip's median car, under --max_class_ratio,
and too_close's real truck reaches 2.6x, so no ratio separates them alone.

What separates them is how much of the frame the box covers:

    yield_runway  the bonnet  100.0% of the frame, all 72 of its instances
    too_close     the truck    71.3% at its closest
    bike_crosswalk  worst      39.1%

Both conditions are required -- over --max_frame_cover AND over half of
--max_class_ratio -- so a genuinely enormous vehicle at normal proportions is
still safe at high coverage, and a badly-proportioned box is still safe unless
it is also engulfing the camera.

THE FLICKER

The bike_crosswalk slab is not caught by either rule above, and cannot be: its
3D box is 5.77 x 2.31 x 1.92 against the real Explorer on four_way at 6.01 x
2.40 x 2.00, with a *smaller* ground footprint. Geometrically it is an
unremarkable car 8.8 m ahead. What it is not is persistent:

                        score   track
    bike_crosswalk slab 0.303   3 frames of 358
    four_way Explorer   0.734   208 of 437
    missing_blinker     0.770   64 of 110
    too_close truck     0.480   136 of 137

A box covering a large part of the frame appears for three frames and vanishes.
Nothing that close goes away that fast -- it would have to cross the whole field
of view in a third of a second -- so a big box on a short track is a fit that
found something for a moment and lost it, which is what a mask sliding off a
vehicle onto tarmac looks like.

The gate is the projected coverage rather than the range, because a range is in
the clip's own units and those differ by a factor of two across this dataset,
while a fraction of the frame means the same thing everywhere. smooth_boxes
--min_track_len already deletes short tracks, but it is blind to size and its
default of 3 is exactly the length this one has; the point here is that how long
a track must be to be believable depends on how big it is.

THE BONNET

The second is the ego's own bonnet lifted into a vehicle -- a box that touches
the bottom edge of the frame and lies entirely below the horizon. The horizon is
where a point straight ahead at infinity projects, so "entirely below" it means
the box has no part further away than the ego itself, which no real object in
front of the camera can manage. A truck you are following crosses the horizon
(on too_close its projected top is 278 px above the frame), so this cannot touch
it. Measured: deer_family 55 dropped, too_close 0, bike_crosswalk 0.

Both rules are per box, not per track: a fit fails on the frames it fails on,
and a track that is a slab on three frames and a car on the rest should lose
three boxes rather than all of them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import box_schema
from lift_frames_to_3d import CLASS_LENGTH_PRIOR
from smooth_boxes import associate

BACKUP = "boxes_3d.prefilter.json"
# Written into the filtered file so a second pass can tell that what it is being
# handed is its own output. Leading underscore, like _manual_3d: every reader in
# this pipeline skips top-level keys starting with one, so it is not a frame.
MARKER = "_ego_filtered"

# A class needs at least this many boxes in the clip before its own median is
# trusted as the reference. Below it the median is one or two boxes, which a
# failed fit can itself dominate -- barrier appears 5 times on too_close.
MIN_CLASS_COUNT = 20


def class_reference(data, frames) -> dict:
    """Typical length per class in this clip's own units.

    A class with enough boxes uses its own median. A rare one is referred to the
    most populous class through CLASS_LENGTH_PRIOR's real-world ratio, so that
    `barrier`, seen five times, is still measured against something this clip
    established rather than against a number in real metres that this clip's
    scale does not share.
    """
    lengths = {}
    for frame in frames:
        entry = data[frame]
        for row, name in zip(box_schema.to_array(entry.get("boxes", [])),
                             entry.get("names", [])):
            lengths.setdefault(name, []).append(float(row[box_schema.LENGTH]))
    if not lengths:
        return {}

    anchor = max(lengths, key=lambda n: len(lengths[n]))
    anchor_median = float(np.median(lengths[anchor]))
    anchor_prior = CLASS_LENGTH_PRIOR.get(anchor)

    reference = {}
    for name, values in lengths.items():
        if len(values) >= MIN_CLASS_COUNT:
            reference[name] = float(np.median(values))
        elif anchor_prior and CLASS_LENGTH_PRIOR.get(name):
            reference[name] = anchor_median * CLASS_LENGTH_PRIOR[name] / anchor_prior
        else:
            reference[name] = anchor_median
    return reference


def short_track_boxes(data, frames, max_len, min_cover) -> set:
    """{(frame, slot)} for big boxes whose track is too short to be real.

    Association is smooth_boxes', for the same reason the rest of this pipeline
    uses it: a second opinion about which boxes are the same object would let
    this delete tracks the smoother believes in.
    """
    doomed = set()
    for track in associate(frames, data, 3.0, 2, max_px=400.0):
        if len(track["idx"]) > max_len:
            continue
        for fi, slot in zip(track["idx"], track["slot"]):
            entry = data[frames[fi]]
            rows = box_schema.to_array(entry.get("boxes", []))
            if slot >= len(rows) or entry.get("camera") is None:
                continue
            aabb = projected_aabb(rows[slot], entry["camera"])
            if aabb is not None and frame_cover(aabb, entry["camera"]) > min_cover:
                doomed.add((frames[fi], slot))
    return doomed


def horizon_row(camera: dict) -> float:
    """The image row a point straight ahead at infinity projects to."""
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    far = ego_to_camera[:3, :3] @ np.array([1e6, 0.0, 0.0]) + ego_to_camera[:3, 3]
    return float((intrinsics @ far)[1] / max(far[2], 1e-9))


def projected_aabb(row, camera):
    """Pixel extent of a box, or None if any corner is behind the camera."""
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1)
                      for sz in (1, -1)], dtype=np.float64) * 0.5
    from gemini_review_boxes import rpy_to_rot     # one rotation helper, not two
    corners = (rpy_to_rot(*row[box_schema.RPY]) @ (signs * row[box_schema.DIMS]).T).T \
        + row[box_schema.CENTER]
    cam = (ego_to_camera[:3, :3] @ corners.T + ego_to_camera[:3, 3:4])
    if (cam[2] < 0.5).any():
        return None
    uv = (intrinsics @ cam)[:2] / cam[2]
    return uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()


def frame_cover(aabb, camera) -> float:
    """Fraction of the image the box's projection actually covers."""
    height, width = camera["image_hw"]
    inner_w = max(0.0, min(aabb[2], width) - max(aabb[0], 0.0))
    inner_h = max(0.0, min(aabb[3], height) - max(aabb[1], 0.0))
    return (inner_w * inner_h) / float(width * height)


def verdict(row, name, camera, reference, max_ratio, max_cover, bonnet,
            cover_min_ratio=2.0, above_horizon=0.0):
    """Why this box should go, or "" to keep it."""
    expected = reference.get(name)
    ratio = (float(row[box_schema.LENGTH]) / expected
             if expected and expected > 0 else None)
    if max_ratio and ratio is not None and ratio > max_ratio:
        return f"{ratio:.1f}x the median {name} in this clip"

    aabb = projected_aabb(row, camera)
    if aabb is None:
        return ""
    cover = frame_cover(aabb, camera)
    # Both, not either: a real vehicle can fill most of the frame when you are
    # about to hit it, and a box can be badly proportioned without engulfing the
    # camera. It is the combination that has no innocent reading.
    if cover > max_cover and ratio is not None and ratio > cover_min_ratio:
        return (f"covers {cover*100:.0f}% of the frame at {ratio:.1f}x "
                f"the median {name}")

    if not bonnet:
        return ""
    height = camera["image_hw"][0]
    # `above_horizon` is how far the box may rise above the horizon and still be
    # called bodywork, as a fraction of the frame. It defaults to 0 -- entirely
    # below -- and MUST be raised per clip, never globally. Measured overshoot:
    #
    #     opposing_crash's face box   0.17 H   artifact
    #     turn_blocker's Mercedes     0.14 H   a real car directly ahead
    #     too_close's truck           0.56 H   a real truck being followed
    #
    # There is no value that catches the first and spares the second, because on
    # this evidence the two are the same shape at the same place on screen. What
    # tells them apart is what is inside the box, which this file cannot see. So
    # the knob exists for the clip where you have looked and know -- on
    # opposing_crash 0.20 clears the face box off 66 frames -- and stays at 0
    # everywhere you have not.
    # Minus, not plus: image rows increase downward, so a box that rises ABOVE
    # the horizon has a SMALLER top row than it. Adding the margin tightened the
    # rule instead of loosening it -- 0.20 dropped 79 boxes on opposing_crash
    # where 0.0 dropped 98.
    limit = horizon_row(camera) - above_horizon * height
    if aabb[3] >= height * 0.98 and aabb[1] > limit:
        note = ("entirely below the horizon" if above_horizon <= 0 else
                f"no more than {above_horizon:.2f} of the frame above the horizon")
        return f"on the ego bonnet (touches the frame bottom, {note})"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True, help="run dir holding boxes_3d.json")
    ap.add_argument("--boxes", default="boxes_3d.json", help="file to filter")
    ap.add_argument("--out", default=None,
                    help="write here instead of in place (in place keeps a backup)")
    ap.add_argument("--max_class_ratio", type=float, default=0.0,
                    help="drop a box longer than this many times its class's median "
                         "length in this clip. 0 (the default) turns the rule OFF: "
                         "it cannot tell a failed fit from a near vehicle in a clip "
                         "whose other detections are distant, and at 4.0 it deleted "
                         "114 real cars on missing_blinker.")
    ap.add_argument("--max_frame_cover", type=float, default=0.85,
                    help="a box covering more than this fraction of the frame, AND "
                         "over --cover_min_ratio in length, is a fit that has "
                         "swallowed the camera (default 0.85)")
    ap.add_argument("--max_flicker_track", type=int, default=4,
                    help="a box covering over --min_flicker_cover of the frame whose "
                         "whole track is this short is a fit that found something for "
                         "a moment and lost it (default 4 frames; 0 = off)")
    ap.add_argument("--min_flicker_cover", type=float, default=0.15,
                    help="how much of the frame a box must cover for the short-track "
                         "rule to apply to it (default 0.15)")
    ap.add_argument("--cover_min_ratio", type=float, default=2.0,
                    help="size gate inside the coverage rule, kept separate from "
                         "--max_class_ratio so that rule can be off while this one "
                         "still refuses to cut a normally-proportioned vehicle "
                         "(default 2.0)")
    ap.add_argument("--bonnet_above_horizon", type=float, default=0.0,
                    help="how far above the horizon a bottom-touching box may still "
                         "reach and count as ego bodywork, as a fraction of frame "
                         "height. 0 (default) = entirely below. Raise it ONLY for a "
                         "clip you have looked at: at 0.20 it also catches a real car "
                         "directly ahead on turn_blocker.")
    ap.add_argument("--no_bonnet", action="store_true",
                    help="skip the ego-bonnet rule, size outliers only")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    source = out_dir / args.boxes
    if not source.exists():
        raise SystemExit(f"error: no {args.boxes} in {out_dir}")
    data = json.loads(source.read_text())
    # Always filter the unfiltered boxes. Stage 3 rewrites boxes_3d.json from the
    # lift on every render, so most of the time this file is fresh -- but when it
    # is not (a second run by hand, a re-render that stopped after this step) the
    # backup beside it is what the thresholds were meant to see. Re-measuring
    # class medians on already-filtered boxes would also move the medians, and a
    # third pass would then cut boxes the second one kept.
    backup_path = out_dir / BACKUP
    already_filtered = bool(data.get(MARKER))
    if already_filtered:
        if not backup_path.exists():
            raise SystemExit(
                f"error: {source.name} is already filtered but {BACKUP} is gone, so\n"
                f"       the boxes it removed cannot be recovered. Re-run stage 3.")
        data = json.loads(backup_path.read_text())
        print(f"{source.name} already filtered; re-filtering {BACKUP} instead")
    frames = sorted(k for k in data if not k.startswith("_"))
    reference = class_reference(data, frames)
    flickers = (short_track_boxes(data, frames, args.max_flicker_track,
                                  args.min_flicker_cover)
                if args.max_flicker_track else set())
    print("clip reference length per class: "
          + ", ".join(f"{n} {v:.2f}" for n, v in sorted(reference.items())))

    dropped, kept, examples = 0, 0, []
    for frame in frames:
        entry = data[frame]
        rows = box_schema.to_array(entry.get("boxes", []))
        names = list(entry.get("names", []))
        camera = entry.get("camera")
        if not len(rows) or camera is None:
            kept += len(rows)
            continue
        doomed = []
        for i, row in enumerate(rows):
            name = names[i] if i < len(names) else "?"
            why = verdict(row, name, camera, reference, args.max_class_ratio,
                          args.max_frame_cover, not args.no_bonnet,
                          args.cover_min_ratio, args.bonnet_above_horizon)
            if not why and (frame, i) in flickers:
                why = "large box on a track too short to be a real object"
            if why:
                doomed.append(i)
                if len(examples) < 8:
                    examples.append(f"    {frame} #{i} {name} "
                                    f"L={row[box_schema.LENGTH]:.2f} -- {why}")
        # Descending, so removing one does not renumber the next. Every list that
        # is parallel to `boxes` moves with it; leaving names behind would
        # silently relabel every box after the hole.
        for i in sorted(doomed, reverse=True):
            for field in ("boxes", "names", "scores", "manual_tracks"):
                if field in entry and i < len(entry[field]):
                    entry[field].pop(i)
        dropped += len(doomed)
        kept += len(rows) - len(doomed)

    print(f"\n{dropped} box(es) dropped, {kept} kept, over {len(frames)} frame(s)")
    for line in examples:
        print(line)
    if dropped > len(examples):
        print(f"    ... and {dropped - len(examples)} more")

    if args.dry_run:
        print("\ndry run: nothing written")
        return
    if args.out:
        target = out_dir / args.out
    else:
        target = source
        # Gated on what `source` WAS, not on what `data` now holds. `data` was
        # reloaded from the backup a moment ago and so never carries the marker;
        # testing it copied the previous pass's *output* over the backup and made
        # the loss permanent, and a third pass then cut three more boxes off its
        # own result. The backup is written once, from the lift's own output.
        if not already_filtered:
            shutil.copyfile(source, backup_path)
            print(f"unfiltered boxes kept as {backup_path.name}")
        else:
            print(f"unfiltered boxes already kept as {backup_path.name}")
    data[MARKER] = True
    box_schema.dump(data, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
