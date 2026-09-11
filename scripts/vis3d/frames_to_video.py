#!/usr/bin/env python3
"""Stitch each clip's rendered frames into an mp4 (H.264, browser-playable).

    python frames_to_video.py                       # every clip in the dataset
    python frames_to_video.py --clips street_race grandma_crash
    python frames_to_video.py --subdir vis3d        # the rasters, not the overlays
    python frames_to_video.py --frame_dir /path/to/frames --name whatever

One video per clip, named <clip><SUFFIX>.mp4 in --out. A clip whose frames are
older than its video is skipped, so re-running after re-rendering two clips
encodes two videos rather than fifty; --force encodes regardless.

A clip that fails -- no frames, an unreadable first frame, ffmpeg exiting
non-zero -- is reported and the run carries on to the next one. Fifty clips is
long enough that stopping on the first fault means losing the other forty-nine,
and the summary at the end names every clip that did not get a video.
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2

BASE = "/fs/nexus-projects/sim2real/aliu/RAP"
DATA_ROOT = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "scripts/vis3d/videos")
SUFFIX = "_gemini"          # kept from the original naming
FPS = 15.0
EXTS = ("*.jpg", "*.jpeg", "*.png")


def find_frames(frame_dir, pattern=None):
    if pattern:
        return sorted(glob.glob(os.path.join(frame_dir, pattern)))
    return sorted(p for ext in EXTS for p in glob.glob(os.path.join(frame_dir, ext)))


def is_stale(frame_dir, output, frame_paths):
    """True when the video needs (re)making: absent, or older than the frames.

    Frame mtimes alone are not enough. trim_frames.py cuts a clip by deleting
    and renumbering files, so the frames that survive keep their original -- and
    now older -- mtimes: a trimmed clip looks untouched by that test even though
    its frame count has changed, and the video keeps showing the frames that
    were cut. The directory's own mtime is what records an add, a delete or a
    rename, so it counts too.
    """
    if not os.path.exists(output):
        return True
    made = os.path.getmtime(output)
    if os.path.getmtime(frame_dir) > made:
        return True
    return any(os.path.getmtime(p) > made for p in frame_paths)


def encode(frame_paths, output, fps):
    """Frames -> mp4. Returns the number written, or raises RuntimeError."""
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"cannot read first frame {frame_paths[0]}")
    h, w = first.shape[:2]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_written = 0
    try:
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"     warning: skipping unreadable frame {path}")
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            proc.stdin.write(frame.tobytes())
            n_written += 1
    except BrokenPipeError:
        # ffmpeg died mid-stream; its own stderr has already said why, and
        # proc.wait() below turns that into the error this raises.
        pass
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    # The original ignored this and printed success whatever ffmpeg did, leaving
    # a truncated or absent mp4 reported as written.
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")
    return n_written


def clip_dirs(args):
    """(clip name, frame dir) for everything in scope, in name order."""
    if args.frame_dir:
        name = args.name or os.path.basename(os.path.normpath(args.frame_dir))
        return [(name, args.frame_dir)]
    dataset_dir = os.path.join(DATA_ROOT, args.dataset)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"error: no dataset at {dataset_dir}")
    names = args.clips or sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)))
    return [(n, os.path.join(dataset_dir, n, args.run, args.subdir)) for n in names]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="CARE_YTB")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip names (default: every clip in the dataset)")
    ap.add_argument("--run", default="1", help="run subdir holding the renders")
    ap.add_argument("--subdir", default="vis3d_overlay",
                    help="render dir inside the run (default vis3d_overlay)")
    ap.add_argument("--frame_dir", default=None,
                    help="one directory of frames, instead of walking a dataset")
    ap.add_argument("--name", default=None,
                    help="output stem for --frame_dir (default: the dir's name)")
    ap.add_argument("--out", default=OUT_DIR, help=f"output dir (default {OUT_DIR})")
    ap.add_argument("--suffix", default=SUFFIX,
                    help=f"appended to the clip name (default {SUFFIX!r})")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--pattern", default=None, help='e.g. "*.jpg"; default all image types')
    ap.add_argument("--force", action="store_true",
                    help="re-encode even when the video is newer than every frame")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="say what would be encoded; write nothing")
    args = ap.parse_args()

    targets = clip_dirs(args)
    os.makedirs(args.out, exist_ok=True)

    made, skipped, failed = [], [], []
    for i, (clip, frame_dir) in enumerate(targets, 1):
        output = os.path.join(args.out, f"{clip}{args.suffix}.mp4")
        head = f"[{i}/{len(targets)}] {clip}"

        if not os.path.isdir(frame_dir):
            print(f"{head}: no {frame_dir}, skipping")
            failed.append((clip, f"no {args.subdir}"))
            continue
        frame_paths = find_frames(frame_dir, args.pattern)
        if not frame_paths:
            print(f"{head}: no frames in {frame_dir}, skipping")
            failed.append((clip, "no frames"))
            continue
        if not args.force and not is_stale(frame_dir, output, frame_paths):
            print(f"{head}: up to date ({len(frame_paths)} frames), skipping")
            skipped.append(clip)
            continue
        if args.dry_run:
            print(f"{head}: would encode {len(frame_paths)} frames -> {output}")
            made.append(clip)
            continue

        try:
            n = encode(frame_paths, output, args.fps)
        except RuntimeError as exc:
            print(f"{head}: FAILED -- {exc}")
            failed.append((clip, str(exc)))
            continue
        print(f"{head}: {n} frames -> {output}")
        made.append(clip)

    verb = "would encode" if args.dry_run else "encoded"
    print(f"\n{verb} {len(made)}, skipped {len(skipped)} up to date, "
          f"{len(failed)} failed")
    if failed:
        for clip, why in failed:
            print(f"  {clip}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
