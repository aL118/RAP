"""Ask whether a clip's point maps describe one rigid scene, and if so, at what scale.

    python calibrate_clip.py --output_dir ../data/CARE_YTB/ice_road/1
    python calibrate_clip.py --all --dataset CARE_YTB       # triage the whole set

Reports only; writes nothing.

WHAT IT MEASURES

Down the ego lane, the ground is flat, level with the vehicle, and a known
height below the camera. So for a strip of ground straight ahead, the point
map's height-below-camera plotted against range should be a horizontal line
sitting at the mount height. Two numbers come out of fitting that line per frame:

  pitch    its slope. The lift assumes 0 (_camera_to_ego is an axis swap plus a
           fixed CAMERA_HEIGHT), so any real value is box float that grows with
           range.
  height   its intercept. Against the mount's true height, the ratio is how far
           the point map's metric scale is off.

WHY PER FRAME, AND WHY THE SPREAD IS THE HEADLINE

A camera bolted to a windscreen has one pitch and one height for the whole clip.
If the per-frame values disagree, the point maps are not a rigid scene, and no
per-clip correction -- no focal length, no scale factor, no road-plane transform
-- can fix them, because there is no single camera to correct to. The spread is
therefore the first thing to read, and the median is worth having only once the
spread is small. Reporting the median alone would hand you a confident number
for a clip that has no such number, which is the specific way this measurement
has misled before.

Measured on CARE_YTB 2026-09-03: 3 clips of 14 came out rigid, 6 plainly did
not, and buick_nearmiss was among the worst (pitch spread 24.6 deg, height
varying 5.4x across 8 frames of one bolted camera).

WHAT IT CANNOT DO

Separate a depth-scale error from a focal-length one: height and object size are
both proportional to z/f, so they are one observable. It reports the ratio. The
argument for blaming the focal is that UniDepth's own fx varies up to 1.9x
within a single clip, which is impossible for a fixed lens.
"""

import argparse
from pathlib import Path

import numpy as np

# The strip of ground to profile: straight ahead, within half a lane of centre,
# from beyond the ego's bonnet out to wherever depth runs out. Metres, not
# pixels, because the point is to sample road and not whatever happens to occupy
# a fixed image region at a given range.
HALF_LANE = 1.8
NEAR, FAR, STEP = 5.0, 45.0, 5.0
ROW_BAND = (0.55, 0.93)      # image rows to draw from, fraction of height
MIN_BIN = 100                # points needed in a range bin to trust its median
MIN_BINS = 3                 # bins needed before a frame yields a pitch at all

# Thresholds for the verdict. A windscreen mount does not change pitch by 6 deg
# or height by half between one frame and the next, so anything past these is the
# depth field moving, not the camera.
RIGID_PITCH_SPREAD = 6.0
RIGID_HEIGHT_SPREAD = 0.6    # metres, peak to peak


def frame_profile(xyz: np.ndarray):
    """(pitch_deg, height_m) from one point map, or None if the road is too thin."""
    height, width = xyz.shape[1:]
    strip = xyz[:, int(ROW_BAND[0] * height):int(ROW_BAND[1] * height), :].reshape(3, -1)
    strip = strip[:, np.isfinite(strip).all(axis=0)]
    strip = strip[:, (np.abs(strip[0]) < HALF_LANE) & (strip[2] > NEAR - 1.0)]

    ranges, drops = [], []
    for low in np.arange(NEAR, FAR, STEP):
        inside = (strip[2] >= low) & (strip[2] < low + STEP)
        if inside.sum() > MIN_BIN:
            ranges.append(low + STEP / 2.0)
            drops.append(float(np.median(strip[1][inside])))   # +y is down = below camera
    if len(ranges) < MIN_BINS:
        return None
    slope, intercept = np.polyfit(ranges, drops, 1)
    return float(np.degrees(np.arctan(slope))), float(intercept)


def measure(xyz_dir: Path, sample: int):
    paths = sorted(xyz_dir.glob("*.xyz.npy"))
    if not paths:
        return []
    if len(paths) > sample:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, sample).astype(int)]
    out = []
    for path in paths:
        profile = frame_profile(np.load(path))
        if profile is not None:
            out.append((path.name.split(".")[0], *profile))
    return out


def verdict(pitches, heights):
    spread = float(np.ptp(pitches))
    # Peak-to-peak metres, not a max/min ratio: a broken frame puts the fitted
    # ground *above* the camera, and a ratio across zero reports six figures of
    # nonsense instead of "1.6 m of disagreement".
    drift = float(np.ptp(heights))
    if spread < RIGID_PITCH_SPREAD and drift < RIGID_HEIGHT_SPREAD:
        return "RIGID", spread, drift
    if spread < RIGID_PITCH_SPREAD * 2:
        return "marginal", spread, drift
    return "NOT RIGID", spread, drift


def report(name: str, rows, camera_height: float, verbose: bool):
    if len(rows) < MIN_BINS:
        print(f"{name:<22} -- only {len(rows)} usable frame(s); nothing to conclude")
        return
    pitches = [r[1] for r in rows]
    heights = [r[2] for r in rows]
    tag, spread, ratio = verdict(pitches, heights)
    median_height = float(np.median(heights))
    print(f"{name:<22} {len(rows):>2} frames  pitch {np.median(pitches):+5.1f} "
          f"(spread {spread:5.1f})  height {median_height:5.2f} m (+-{ratio:4.2f})  "
          f"scale {median_height / camera_height:5.2f}x  {tag}")
    if verbose:
        for frame, pitch, height in rows:
            print(f"    {frame}  pitch {pitch:+6.1f} deg   height {height:6.2f} m")
        if tag == "RIGID":
            print(f"    -> one camera fits this clip. Correcting it by "
                  f"{median_height / camera_height:.2f}x is meaningful.")
        else:
            print( "    -> the per-frame values disagree by more than a bolted camera can.\n"
                   "       No single scale or pitch describes this clip; a per-clip\n"
                   "       correction would fix the average and leave the variation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", help="run dir holding samples-pseudodepth/")
    ap.add_argument("--all", action="store_true", help="triage every clip in --dataset")
    ap.add_argument("--dataset", default="CARE_YTB")
    ap.add_argument("--run", default="1")
    ap.add_argument("--camera_height", type=float, default=1.5,
                    help="the mount's real height above the road, metres (car 1.2-1.5)")
    ap.add_argument("--sample", type=int, default=8, help="frames to measure per clip")
    args = ap.parse_args()

    if args.all:
        data = Path(__file__).resolve().parents[1] / "data" / args.dataset
        for clip in sorted(p for p in data.iterdir() if p.is_dir()):
            xyz_dir = clip / args.run / "samples-pseudodepth" if args.run \
                else clip / "samples-pseudodepth"
            if xyz_dir.is_dir():
                report(clip.name, measure(xyz_dir, args.sample), args.camera_height, False)
        return
    if not args.output_dir:
        ap.error("give --output_dir, or --all to triage a dataset")
    xyz_dir = Path(args.output_dir).resolve() / "samples-pseudodepth"
    if not xyz_dir.is_dir():
        raise SystemExit(f"error: no point maps at {xyz_dir}")
    # The run dir is usually "1", which names nothing; the clip is its parent.
    path = Path(args.output_dir).resolve()
    name = f"{path.parent.name}/{path.name}" if path.name.isdigit() else path.name
    report(name, measure(xyz_dir, args.sample), args.camera_height, True)


if __name__ == "__main__":
    main()
