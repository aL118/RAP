"""Apply hand-made 3D box corrections to a run's boxes_3d.json.

    python apply_manual_boxes_3d.py --output_dir ../data/CARE_YTB/opposing_crash/2hz

Corrections are drawn in annotate_boxes.py --boxes3d and land in
manual_boxes_3d.json beside boxes_3d.json. They are kept in their own file and
re-applied, rather than written into boxes_3d.json once, for the same reason
manual_boxes.json is: stage 3 rewrites boxes_3d.json from scratch every time it
runs, so an edit made in place survives exactly until the next re-lift and then
vanishes without saying so.

WHY THE 2 Hz RUN AND NOT THE 10 Hz ONE

The 2 Hz run is what is delivered -- what the navsim export reads and what
training sees -- so a correction there is final and nothing downstream can undo
it. It is also a fifth of the frames, which for hand work is the difference
between a tool being used and not. Correcting the 10 Hz run instead would mean
the edit is re-derived through subsampling rather than authoritative, and a
correction made at 2 Hz cannot be pushed back to 10 Hz anyway without inventing
the four intervening frames it says nothing about.

THREE OPERATIONS

  delete   drops the lifted box nearest `at`, within `radius` metres. For a
           false positive: the ego's own bonnet, a hedge, a road surface.
  replace  drops that box and puts `box` in its place. For a box on a real
           object at the wrong depth, size or heading.
  add      inserts `box` unconditionally. For an object the detector missed.

delete and replace are anchored to a POSITION rather than an index, because an
index means nothing across a re-lift: the boxes are rebuilt, in a different
order, in slightly different places. A position with a tolerance still finds the
same object, and says so when it cannot -- an edit that matches nothing is
reported rather than dropped in silence, since that is the signal that the lift
moved under the annotation and it needs looking at again.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import box_schema

PRISTINE = "boxes_3d.lifted.json"
MERGED = "boxes_3d.json"
MANUAL = "manual_boxes_3d.json"

# How far a lifted box's centre may sit from an edit's anchor and still be taken
# to be the same object, in metres, when the edit does not carry its own radius.
# Generous, because the whole reason a box is being corrected is often that the
# lift put it in the wrong place -- but far below the spacing between distinct
# vehicles in these scenes.
DEFAULT_RADIUS = 3.0


def apply_edits(boxes_by_frame: dict, edits: dict):
    """Applies the edit list to a lifted boxes_3d structure, in place.

    Returns (applied, unmatched) counts per operation for reporting.
    """
    applied = {"delete": 0, "replace": 0, "add": 0}
    unmatched = []

    for frame, frame_edits in edits.items():
        entry = boxes_by_frame.get(frame)
        if entry is None:
            unmatched.extend((frame, e.get("op", "?"), "no such frame") for e in frame_edits)
            continue

        # Anchored ops first, and all of them resolved against the ORIGINAL box
        # list before anything is removed: resolving one at a time would let the
        # first deletion shift the indices the second was matched on.
        claimed = set()
        resolved = []
        for edit in frame_edits:
            op = edit.get("op")
            if op == "add":
                continue
            anchor = np.asarray(edit["at"], dtype=float)
            radius = float(edit.get("radius", DEFAULT_RADIUS))
            best, best_dist = None, radius
            for i, box in enumerate(entry["boxes"]):
                if i in claimed:
                    continue
                dist = float(np.linalg.norm(box_schema.from_any(box)[box_schema.CENTER] - anchor))
                if dist < best_dist:
                    best, best_dist = i, dist
            if best is None:
                unmatched.append((frame, op, f"nothing within {radius:.1f} m of "
                                              f"({anchor[0]:.1f}, {anchor[1]:.1f}, {anchor[2]:.1f})"))
                continue
            claimed.add(best)
            resolved.append((best, edit))

        # Descending, so removing one does not renumber the next.
        for index, edit in sorted(resolved, key=lambda r: -r[0]):
            for field in ("boxes", "names", "scores", "manual_tracks"):
                if field in entry and index < len(entry[field]):
                    entry[field].pop(index)
            if edit["op"] == "replace":
                _append(entry, edit["box"], edit.get("name", "car"),
                        float(edit.get("score", 1.0)))
                applied["replace"] += 1
            else:
                applied["delete"] += 1

        for edit in frame_edits:
            if edit.get("op") == "add":
                _append(entry, edit["box"], edit.get("name", "car"),
                        float(edit.get("score", 1.0)))
                applied["add"] += 1

    return applied, unmatched


def _append(entry: dict, box, name: str, score: float):
    entry["boxes"].append(box_schema.to_dict(box))
    entry["names"].append(name)
    entry["scores"].append(score)
    # Kept parallel: smooth_boxes and the export index these lists together, and
    # a short manual_tracks is padded with None rather than missing entries.
    if "manual_tracks" in entry:
        entry["manual_tracks"].append(None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True, help="run dir holding boxes_3d.json")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    manual_path, merged_path = out_dir / MANUAL, out_dir / MERGED
    pristine_path = out_dir / PRISTINE
    if not manual_path.exists():
        print(f"no {MANUAL} in {out_dir}; nothing to apply")
        return
    if not merged_path.exists():
        raise SystemExit(f"error: no {MERGED} in {out_dir}")

    manual = json.loads(manual_path.read_text())
    edits = manual.get("edits", {})

    # Edits from `gemini_review_boxes.py --in_boxes <other file>` are anchored to
    # that file, not to the lift. This script applies onto the lift, where those
    # anchors point at whatever happens to be nearest -- a different object, or
    # nothing. That is silent corruption, so it stops instead.
    anchored_to = manual.get("_anchored_to")
    if anchored_to and anchored_to != MERGED:
        raise SystemExit(
            f"error: {MANUAL} holds edits anchored to {anchored_to}, not {MERGED}.\n"
            f"       They were written by a review of that file and only mean\n"
            f"       something when applied to it. Re-run the review without\n"
            f"       --in_boxes, or apply them with:\n"
            f"         python gemini_review_boxes.py --from_findings --apply \\\n"
            f"             --in_boxes {anchored_to} --out_boxes {anchored_to} ...")

    # Same rule as apply_manual_boxes.py: always start from the lift's own
    # output. A file that already carries corrections is a previous merge, and
    # re-applying on top of it would delete a second box for every delete.
    current = json.loads(merged_path.read_text())
    if current.get("_manual_3d"):
        if not pristine_path.exists():
            raise SystemExit(
                f"error: {MERGED} already holds corrections but {PRISTINE} is gone, so the\n"
                f"       lift's own boxes cannot be recovered. Re-run stage 3.")
        boxes_by_frame = json.loads(pristine_path.read_text())
        source = f"{PRISTINE} ({MERGED} is a previous merge)"
    else:
        boxes_by_frame = current
        source = MERGED
        if not args.dry_run:
            shutil.copyfile(merged_path, pristine_path)
    boxes_by_frame.pop("_manual_3d", None)
    print(f"lifted boxes read from {source}")

    applied, unmatched = apply_edits(boxes_by_frame, edits)
    total = sum(len(v) for v in edits.values())
    print(f"{total} edit(s) over {len(edits)} frame(s): "
          f"{applied['delete']} deleted, {applied['replace']} replaced, {applied['add']} added")
    if unmatched:
        print(f"\n{len(unmatched)} edit(s) matched nothing -- the lift has moved under them:")
        for frame, op, why in unmatched[:12]:
            print(f"    {frame}  {op:<8} {why}")
        if len(unmatched) > 12:
            print(f"    ... and {len(unmatched) - 12} more")
        print("    Re-open the clip in annotate_boxes.py --boxes3d and redo these; the\n"
              "    anchors are positions, so a re-lift that moved a box far enough breaks them.")

    if args.dry_run:
        print("dry run: nothing written")
        return
    boxes_by_frame["_manual_3d"] = True
    tmp = merged_path.with_suffix(".json.tmp")
    box_schema.dump(boxes_by_frame, tmp)
    tmp.replace(merged_path)
    print(f"wrote {merged_path}")
    print(f"re-render the overlays:\n  cd visualization && python raster_frames.py "
          f"--output_dir {out_dir} --frames_dir {out_dir.parent}/frames_2hz")


if __name__ == "__main__":
    main()
