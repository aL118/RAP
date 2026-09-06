"""Take a clip processed at a high rate down to a lower one, after the fact.

    python subsample_clip.py --clip_dir ../data/CARE_YTB/buick_nearmiss_test \
        --run 1 --stride 5

Why process at 10 Hz and keep 2 Hz, rather than extracting 2 Hz to begin with:
everything in this pipeline that reasons across time gets easier as the frames
get closer together, and none of it gets harder.

  association   smooth_boxes gates on how far a box's projected centre moves
                between frames. At 2 Hz a vehicle being passed crosses hundreds
                of pixels in half a second and the gate has to be opened wide
                enough to also admit the wrong object; at 10 Hz it moves a fifth
                as far and the gate can be tight.
  smoothing     the yaw and extent medians have five times the samples over the
                same stretch of road, so a spike has to persist five times
                longer to survive them.
  depth drift   UniDepth's metric scale wanders with elapsed time rather than
                jittering per frame -- measured on buick_nearmiss, the
                frame-to-frame scale spread is 1.23x at 0.1 s and 1.99x at 0.5 s.
                Tracks built at 10 Hz therefore span less drift per link.

The cost is five times the stage-1 and stage-2 compute, which is why this exists
as an experiment to be measured rather than a default.

Nothing is recomputed here. The boxes are the ones already lifted and smoothed at
the high rate; this selects every stride-th frame and renumbers it consecutively,
so the result is directly comparable, frame for frame, with the same clip
extracted at the low rate in the first place.
"""

import argparse
import json
import shutil
from pathlib import Path


def frame_list(frames_dir: Path):
    return sorted(p for p in frames_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip_dir", required=True, help="clip directory holding frames/ and the run")
    ap.add_argument("--run", default="1", help="run subdir the high-rate outputs are in")
    ap.add_argument("--frames", default="frames",
                    help="subdir holding the HIGH-RATE frames the run was built from. Not "
                         "always 'frames': a clip processed at 10 Hz alongside its original "
                         "2 Hz one keeps them in frames_10hz/, and pointing this at the 2 Hz "
                         "set would silently subsample the wrong sequence.")
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame (10 Hz -> 2 Hz = 5)")
    ap.add_argument("--phase", type=int, default=0,
                    help="which of the stride frames to keep. Changing it shifts the kept "
                         "set by a fraction of the source period, which is the only way to "
                         "line the output up with a differently-phased extraction.")
    ap.add_argument("--out_frames", default=None, help="default: frames_<rate>hz next to frames/")
    ap.add_argument("--out_run", default=None, help="default: <run>_sub")
    ap.add_argument("--copy", action="store_true",
                    help="copy the frames instead of symlinking them. Links are the default "
                         "because the high-rate frames are staying on disk anyway and a "
                         "second copy of 157 jpgs earns nothing.")
    args = ap.parse_args()

    clip_dir = Path(args.clip_dir).resolve()
    frames_dir = clip_dir / args.frames
    run_dir = clip_dir / args.run if args.run else clip_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"error: no frames at {frames_dir}")
    boxes_path = run_dir / "boxes_3d.json"
    if not boxes_path.exists():
        raise SystemExit(f"error: no boxes_3d.json in {run_dir}; run the pipeline first")

    frames = frame_list(frames_dir)
    # Names, not counts. boxes_3d.json holds an entry only for a frame that has
    # something in it -- boxes, lanes or traffic lights -- so a clip on unmarked
    # road with little traffic legitimately covers a fraction of its frames, and
    # a count check rejects it as if --frames were wrong. What actually goes
    # wrong is pointing at a different extraction, and then the names disagree.
    boxed = set(json.loads(boxes_path.read_text()))
    stray = boxed - {f.name for f in frames}
    if stray:
        raise SystemExit(
            f"error: {len(stray)} of {len(boxed)} frame(s) named in {args.run}/boxes_3d.json "
            f"are not in\n       {frames_dir.name}/ (e.g. {sorted(stray)[0]}). --frames is "
            f"pointing at a different\n       extraction than the run was built from.")
    if boxed:
        print(f"  {len(boxed)} of {len(frames)} frame(s) carry boxes, lanes or lights")
    if args.phase >= args.stride:
        raise SystemExit(f"error: --phase must be below --stride ({args.stride})")
    kept = frames[args.phase::args.stride]
    if not kept:
        raise SystemExit("error: stride and phase selected no frames")

    out_frames = Path(args.out_frames) if args.out_frames else clip_dir / "frames_sub"
    out_run = Path(args.out_run) if args.out_run else clip_dir / f"{args.run}_sub"
    if not out_frames.is_absolute():
        out_frames = clip_dir / out_frames
    if not out_run.is_absolute():
        out_run = clip_dir / out_run
    for directory in (out_frames, out_run):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    # Renumbered consecutively rather than keeping the source indices: a
    # downstream stage that infers a timestamp from the frame number would
    # otherwise read this clip as 10 Hz with four frames missing out of five.
    renamed = {}
    for i, source in enumerate(kept):
        target = out_frames / f"{i:06d}{source.suffix}"
        renamed[source.name] = target.name
        if args.copy:
            shutil.copyfile(source, target)
        else:
            target.symlink_to(source.resolve())

    boxes = json.loads(boxes_path.read_text())
    subsampled = {renamed[name]: entry for name, entry in boxes.items() if name in renamed}
    missing = len(renamed) - len(subsampled)
    (out_run / "boxes_3d.json").write_text(json.dumps(subsampled))

    poses = run_dir / "ego_poses.txt"
    if poses.exists():
        # One line per frame, in frame order -- selected with the same stride so
        # the pose track still lines up with the boxes it was solved alongside.
        lines = poses.read_text().splitlines()
        if len(lines) == len(frames):
            (out_run / "ego_poses.txt").write_text(
                "\n".join(lines[args.phase::args.stride]) + "\n")
        else:
            print(f"note: ego_poses.txt has {len(lines)} lines for {len(frames)} frames; "
                  f"not subsampling it")

    print(f"{len(frames)} frames -> {len(kept)} (every {args.stride}th, phase {args.phase})")
    if missing:
        print(f"  {missing} kept frame(s) had no entry in boxes_3d.json")
    print(f"  frames: {out_frames}")
    print(f"  run:    {out_run}")
    print(f"\nrender the overlays with:\n"
          f"  cd visualization && python raster_frames.py --output_dir {out_run} "
          f"--frames_dir {out_frames}")


if __name__ == "__main__":
    main()
