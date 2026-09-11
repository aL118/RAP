"""Cut a processed clip down to a frame range, in place, taking every artifact keyed to a frame with it.

    # see what would happen -- this is the default, nothing is written
    python trim_frames.py --clip_dir ../../data/CARE_YTB/close_nightcrash --run 1 --start 60 --end 140

    # do it
    python trim_frames.py --clip_dir ../../data/CARE_YTB/close_nightcrash --run 1 --start 60 --end 140 --apply

--start/--end are inclusive and are stated in the clip's ORIGINAL numbering, so
the range you read off a vis3d_overlay frame is the range you type. The kept
frames are then renumbered consecutively from 000000, because several stages
downstream read the frame number as a clock: export_navsim_logs.py indexes
ego_poses by a frame's own number (000123.jpg -> ego_poses[123]) rather than by
position in a list, so a clip whose frames start at 60 would be read as one
starting at t=6s with 60 frames missing. Renumbering keeps that mapping honest.
Pass --keep-numbering to leave the original indices alone.

What gets trimmed, and how it is recognised:

  NNNNNN.*            every per-frame file anywhere under the run dir, found by
  (recursive)         name, not by a hardcoded list of directories -- frames/,
                      vis/, vis3d/, vis3d_overlay/, drivable_masks/,
                      lane_masks/, and samples-pseudodepth/ (NNNNNN.K.npy and
                      NNNNNN.xyz.npy both carry the index before the first dot).
                      A directory added to the pipeline later is picked up for
                      free as long as it names files after their frame.
  review_NNNNNN.jpg   gemini_review/, same thing behind a prefix.
  boxes_3d.json       dict keyed by frame name. Keys that are not frame names
  gemini_boxes_3d     (_ego_filtered, _manual_3d) are pipeline metadata and are
                      copied through untouched -- dropping them would silently
                      re-arm filters that have already run.
  manual_boxes_3d     the frame dict is nested one level down, under "edits".
  mask_results_preds  a flat list of records, each tagged with "frame".
  findings.json       gemini_review's, keyed by frame name.
  ego_poses.txt       one KITTI-format 3x4 camera-to-world row per frame, and
                      line i IS frame i -- so the kept lines are selected by
                      frame number, not by slicing positions. They are then
                      rebased onto the first kept frame (T_0^-1 @ T_i), which
                      restores the "starts from identity" invariant that
                      load_ego_poses documents. Relative motion is unchanged;
                      --no-rebase-poses leaves them in the original world frame.
  info.json           event_frames holds absolute frame INDICES, so it is
                      remapped alongside everything else. An event frame that
                      falls outside the kept range is dropped with a warning:
                      trimming away the event you are keeping the clip for is
                      almost always a mistake in --start/--end.

What gets deleted rather than trimmed (--keep-derived to opt out): the
whole-clip derived artifacts that have no per-frame structure to edit --
navsim_logs/*.pkl (a token linked list over a subsampled 47-record view, not a
frame-indexed array), its paired sensor_blobs/ (frame-named, but symlinks into
frames/ that renumbering would strand -- see DERIVED), openvo/results/,
ego_trajectory.png, and gemini_review/report.md. They are stale the instant a
frame is dropped, and a stale navsim pkl is worse than a missing one because its
frame_idx still looks valid. The commands that rebuild them are printed at the
end.

The clip's source .mp4 is never touched: it is the original download and the
only way back to the frames this throws away.
"""

# Clips queued for trimming, as ranges in their original numbering:
#     poland_slip: 0-180
#     street_race: 0-180
#     turn_blocker: 0-200

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

# The index is whatever leads the basename, before the first dot -- "000000.jpg"
# and "000000.xyz.npy" both resolve to 0. Anchored at both ends so a file that
# merely starts with digits (a hypothetical "000000_summary.txt") does not get
# swept up and renamed as if it belonged to a frame.
FRAME_RE = re.compile(r"^(\d{6})(\..+)$")
REVIEW_RE = re.compile(r"^(review_)(\d{6})(\..+)$")

# Derived from the whole clip; regenerated, never edited. Paths are relative to
# the run dir; a directory means the whole subtree.
#
# sensor_blobs/ is here rather than in the per-frame sweep even though its
# contents ARE named after frames: export_navsim_logs.py writes it and
# navsim_logs/*.pkl as one artifact, subsampled to every 5th frame (47 of 231 on
# close_nightcrash), and its entries are absolute symlinks into frames/. Renaming
# those links while their targets are being renumbered leaves 47 dangling
# pointers attached to a pkl this script is deleting anyway. Both go, together.
DERIVED = ["navsim_logs", "sensor_blobs", "openvo", "ego_trajectory.png",
           "gemini_review/report.md"]

BOX_JSONS = ["boxes_3d.json", "gemini_boxes_3d.json"]
NESTED_JSONS = ["manual_boxes_3d.json"]          # frame dict one level down
RECORD_JSONS = ["mask_results_preds.json"]       # list of {"frame": ...}
REVIEW_JSONS = ["gemini_review/findings.json"]   # dict keyed by frame name


def is_frame_name(key: str) -> bool:
    return bool(FRAME_RE.match(key))


def frame_index(name: str):
    m = FRAME_RE.match(name)
    return int(m.group(1)) if m else None


class Plan:
    """Everything the run would do, collected before any of it happens.

    Built in full first so that --apply is a replay of exactly what --dry-run
    printed, and so a clip that is going to fail a consistency check fails
    before the first file is unlinked rather than halfway through.
    """

    def __init__(self):
        self.renames = []    # (src, dst)
        self.deletes = []    # path
        self.rewrites = []   # (path, note)
        self.warnings = []

    def rename(self, src, dst):
        if src != dst:
            self.renames.append((src, dst))

    def delete(self, path):
        self.deletes.append(path)

    def rewrite(self, path, note):
        self.rewrites.append((path, note))

    def warn(self, message):
        self.warnings.append(message)


def remap_frame_dict(data: dict, keep: dict) -> tuple:
    """Filters a frame-keyed dict to the kept frames and renames the keys.

    Returns (new_dict, n_dropped). Non-frame keys are preserved in place.
    """
    out, dropped = {}, 0
    for key, value in data.items():
        if not is_frame_name(key):
            out[key] = value        # metadata (_ego_filtered, _manual_3d)
            continue
        if key in keep:
            out[keep[key]] = value
        else:
            dropped += 1
    return out, dropped


def rebase_poses(rows: np.ndarray) -> np.ndarray:
    """Re-expresses 3x4 camera-to-world rows so the first one is identity."""
    mats = np.tile(np.eye(4), (len(rows), 1, 1))
    mats[:, :3, :4] = rows
    return (np.linalg.inv(mats[0]) @ mats)[:, :3, :4]


def build_plan(args, clip_dir: Path, frames_dir: Path, run_dir: Path) -> tuple:
    plan = Plan()

    frames = sorted(p for p in frames_dir.iterdir() if FRAME_RE.match(p.name))
    if not frames:
        raise SystemExit(f"error: no NNNNNN.* frames in {frames_dir}")
    indices = [frame_index(p.name) for p in frames]

    end = args.end if args.end is not None else max(indices)
    if args.start > end:
        raise SystemExit(f"error: --start {args.start} is above --end {end}")
    kept_idx = [i for i in indices if args.start <= i <= end]
    if not kept_idx:
        raise SystemExit(
            f"error: frames {args.start}-{end} selected nothing; this clip covers "
            f"{min(indices)}-{max(indices)}")
    if end > max(indices):
        plan.warn(f"--end {end} is past the last frame ({max(indices)}); "
                  f"nothing is trimmed off the tail")
    if len(kept_idx) == len(indices):
        plan.warn("the range covers every frame: nothing would be dropped")

    # old index -> new index. The whole rest of the run keys off this.
    offset = args.start if not args.keep_numbering else 0
    index_map = {i: i - offset for i in kept_idx}

    # Frame filenames, for remapping the JSON keys. Extension is carried across
    # from the source so a png-frame clip maps to png.
    keep_names = {}
    for path in frames:
        i = frame_index(path.name)
        if i in index_map:
            keep_names[path.name] = f"{index_map[i]:06d}{FRAME_RE.match(path.name).group(2)}"

    # --- per-frame files, anywhere under the clip -------------------------
    # Renames are emitted in ascending index order, which is what makes an
    # in-place renumber safe without a scratch pass: every target index is at or
    # below its source, so the slot it lands in was either already vacated by
    # the delete pass or by the rename before it.
    # Deduplicated by path: with --run '' the run dir IS the clip dir, so
    # frames_dir sits inside it and a naive two-root walk would plan every
    # frame's rename twice -- the second of which would fail on a missing file.
    # Anything living under a derived path is that artifact's business, not the
    # per-frame sweep's: the whole subtree is deleted below, and planning a
    # rename inside it would target a file that no longer exists by the time the
    # renames run.
    derived_roots = [run_dir / rel for rel in DERIVED]

    def under_derived(path):
        return any(path == root or root in path.parents for root in derived_roots)

    seen = {}
    for root in (frames_dir, run_dir):
        for path in root.rglob("*"):
            if not path.is_file() or path in seen or under_derived(path):
                continue
            m, prefix = FRAME_RE.match(path.name), ""
            if not m:
                m = REVIEW_RE.match(path.name)
                if not m:
                    continue
                prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
            else:
                digits, suffix = m.group(1), m.group(2)
            seen[path] = (int(digits), path, prefix, suffix)
    per_frame = list(seen.values())

    # A per-frame file that is a symlink into the clip is a link whose target is
    # about to be renumbered out from under it. sensor_blobs/ was the only such
    # producer when this was written and is deleted wholesale above, so this is
    # here to catch the next one rather than any case that exists today.
    dangling = [p for _, p, _, _ in per_frame
                if p.is_symlink() and clip_dir in Path(p.resolve()).parents]
    if dangling:
        plan.warn(f"{len(dangling)} per-frame file(s) are symlinks into this clip "
                  f"(e.g. {dangling[0].relative_to(clip_dir)}); renumbering their "
                  f"targets will break them")

    for i, path, prefix, suffix in sorted(per_frame, key=lambda t: t[0]):
        if i in index_map:
            plan.rename(path, path.with_name(f"{prefix}{index_map[i]:06d}{suffix}"))
        else:
            plan.delete(path)

    # --- frame-keyed json -------------------------------------------------
    edits = {}
    for rel in BOX_JSONS + REVIEW_JSONS:
        path = run_dir / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        stray = {k for k in data if is_frame_name(k)} - {p.name for p in frames}
        if stray:
            raise SystemExit(
                f"error: {rel} names {len(stray)} frame(s) that are not in "
                f"{frames_dir.name}/ (e.g. {sorted(stray)[0]}).\n"
                f"       --frames is pointing at a different extraction than the run.")
        new, dropped = remap_frame_dict(data, keep_names)
        edits[path] = new
        kept = sum(1 for k in new if is_frame_name(k))
        plan.rewrite(path, f"{dropped} frame entries dropped, {kept} kept")

    for rel in NESTED_JSONS:
        path = run_dir / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        dropped = 0
        new = {}
        for key, value in data.items():
            # "edits" and anything else shaped like a frame dict; other keys ride through.
            if isinstance(value, dict) and any(is_frame_name(k) for k in value):
                new[key], n = remap_frame_dict(value, keep_names)
                dropped += n
            else:
                new[key] = value
        edits[path] = new
        plan.rewrite(path, f"{dropped} frame entries dropped")

    for rel in RECORD_JSONS:
        path = run_dir / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        new = []
        for record in data:
            name = record.get("frame")
            if name in keep_names:
                record = dict(record, frame=keep_names[name])
                new.append(record)
        edits[path] = new
        plan.rewrite(path, f"{len(data) - len(new)} of {len(data)} records dropped")

    # --- ego poses --------------------------------------------------------
    poses_path = run_dir / "ego_poses.txt"
    if poses_path.exists():
        lines = poses_path.read_text().split()
        rows = np.array(lines, dtype=np.float64).reshape(-1, 3, 4)
        if len(rows) != len(frames):
            plan.warn(f"ego_poses.txt has {len(rows)} poses for {len(frames)} frames; "
                      f"leaving it alone -- regenerate it with stage 4")
        elif max(kept_idx) >= len(rows):
            plan.warn(f"ego_poses.txt is short of frame {max(kept_idx)}; leaving it alone")
        else:
            # By frame number, not by slice: line i is frame i, and that holds
            # even if the extraction ever leaves a gap in the numbering.
            picked = rows[kept_idx]
            if not args.no_rebase_poses:
                picked = rebase_poses(picked)
            edits[poses_path] = picked
            plan.rewrite(poses_path, f"{len(rows)} -> {len(picked)} poses"
                         + ("" if args.no_rebase_poses else ", rebased onto the first kept frame"))

    # --- info.json --------------------------------------------------------
    info_path = clip_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        events = info.get("event_frames")
        if isinstance(events, list) and events:
            remapped = [index_map[e] for e in events if e in index_map]
            lost = [e for e in events if e not in index_map]
            if lost:
                plan.warn(f"info.json event_frames {lost} fall outside {args.start}-{end} "
                          f"and would be dropped -- check the range covers the event")
            if remapped != events:
                edits[info_path] = dict(info, event_frames=remapped)
                plan.rewrite(info_path, f"event_frames {events} -> {remapped}")

    # --- derived ----------------------------------------------------------
    if not args.keep_derived:
        for rel in DERIVED:
            path = run_dir / rel
            if path.exists():
                plan.delete(path)

    source_video = next((p for p in clip_dir.glob("*.mp4")), None)
    if source_video is not None:
        plan.warn(f"{source_video.name} still covers all {len(frames)} original frames "
                  f"(left alone: it is the only way back)")

    return plan, kept_idx, indices, edits, index_map


def apply_plan(plan: Plan, edits: dict):
    for path in plan.deletes:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for src, dst in plan.renames:
        src.rename(dst)
    for path, payload in edits.items():
        if isinstance(payload, np.ndarray):
            np.savetxt(path, payload.reshape(len(payload), 12))
        else:
            path.write_text(json.dumps(payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip_dir", required=True, help="clip directory holding frames/ and the run")
    ap.add_argument("--run", default="1", help="run subdir ('' = the clip dir itself)")
    ap.add_argument("--frames", default="frames", help="subdir holding the frames")
    ap.add_argument("--start", type=int, required=True,
                    help="first frame to KEEP, in the clip's original numbering (inclusive)")
    ap.add_argument("--end", type=int, default=None,
                    help="last frame to KEEP, inclusive (default: the last frame)")
    ap.add_argument("--apply", action="store_true",
                    help="actually do it. Without this the plan is printed and nothing is written.")
    ap.add_argument("--keep-numbering", action="store_true",
                    help="leave the kept frames at their original indices instead of "
                         "renumbering from 000000. See the note above about frame "
                         "numbers being read as a clock.")
    ap.add_argument("--keep-derived", action="store_true",
                    help="do not delete navsim_logs/, openvo/, ego_trajectory.png and "
                         "report.md. They will be inconsistent with the trimmed clip.")
    ap.add_argument("--no-rebase-poses", action="store_true",
                    help="keep ego_poses.txt in its original world frame instead of "
                         "rebasing so the first kept frame is identity.")
    args = ap.parse_args()

    clip_dir = Path(args.clip_dir).resolve()
    frames_dir = clip_dir / args.frames
    run_dir = clip_dir / args.run if args.run else clip_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"error: no frames at {frames_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"error: no run dir at {run_dir}")

    plan, kept_idx, indices, edits, index_map = build_plan(args, clip_dir, frames_dir, run_dir)

    end = args.end if args.end is not None else max(indices)
    print(f"{clip_dir.name}: keep frames {args.start}-{end} "
          f"({len(kept_idx)} of {len(indices)}), drop {len(indices) - len(kept_idx)}")
    if not args.keep_numbering and args.start:
        print(f"  renumbering {args.start:06d} -> 000000 "
              f"(shift of {args.start})")

    print(f"\n  delete {len(plan.deletes)} path(s)")
    for path in plan.deletes[:4]:
        print(f"    {path.relative_to(clip_dir)}")
    if len(plan.deletes) > 4:
        print(f"    ... and {len(plan.deletes) - 4} more")

    print(f"  rename {len(plan.renames)} file(s)")
    for src, dst in plan.renames[:4]:
        print(f"    {src.relative_to(clip_dir)} -> {dst.name}")
    if len(plan.renames) > 4:
        print(f"    ... and {len(plan.renames) - 4} more")

    print(f"  rewrite {len(plan.rewrites)} file(s)")
    for path, note in plan.rewrites:
        print(f"    {path.relative_to(clip_dir)}: {note}")

    for message in plan.warnings:
        print(f"\n  warning: {message}")

    if not args.apply:
        print("\nDry run. Nothing was written -- re-run with --apply.")
        return

    apply_plan(plan, edits)
    print(f"\nTrimmed {clip_dir.name} to {len(kept_idx)} frames.")
    if not args.keep_derived:
        run_arg = f"RUN={args.run}" if args.run else "RUN="
        print("Regenerate what was deleted with:\n"
              f"  cd {Path(__file__).resolve().parent} && "
              f"DATASET={clip_dir.parent.name} VIDEO={clip_dir.name} {run_arg} \\\n"
              f"    RUN_STAGE4_VO=1 EXPORT_NAVSIM=1 ./rerun_vis3d.sh")


if __name__ == "__main__":
    main()
