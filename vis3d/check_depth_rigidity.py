"""Is a clip's depth field scale-consistent between frames? Measured off the plane.

    python check_depth_rigidity.py --output_dir ../data/CARE_YTB/buick_nearmiss/1

calibrate_clip.py reads the scale off a road-plane fit, which is exactly the
measurement that misled this project once before: on beepbeep the fitted road
height wandered 0.26-5.49 m and it was the *fit* moving, not the data -- a thin
distant strip has no lever arm in the height direction, so the fit trades height
against pitch and lands anywhere. Height correlated -0.80 with the depth extent
of the patch it was fitted to, and the frame-to-frame scale measured
independently was 0.9964 with no trend.

This is that independent measurement. It never fits a plane.

  1. Corners in frame i, tracked to i+1 by Lucas-Kanade.
  2. Each surviving correspondence read out of both frames' point maps, giving
     the same physical points in two camera frames.
  3. RANSAC-Umeyama for the similarity that maps one set onto the other.

If the depth field is consistent, the two clouds differ by the ego's own motion
-- a rigid transform -- and the fitted scale is 1.0. A scale away from 1.0 is
the point maps changing size between frames, which no per-clip correction can
undo. RANSAC because the scene contains moving vehicles, whose points do not
follow the ego's rigid motion and would drag a least-squares fit.

Read the SPREAD of the per-pair scales, not their median: a clip whose depth
breathes 0.8x one frame and 1.3x the next has a median near 1.0 and is not
consistent at all.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

MAX_CORNERS = 1200
CORNER_QUALITY = 0.01
CORNER_MIN_DIST = 12
# 2 Hz is half a second of ego motion per pair, which is a large displacement
# for LK: a small window and few pyramid levels lose most of the near field,
# where the parallax that constrains scale actually is.
LK_WINDOW = (31, 31)
LK_LEVELS = 5
FB_TOL_PX = 1.5               # forward-backward check; a track that does not
                              # return to where it started is not a track
MIN_PAIR_POINTS = 40          # correspondences needed before a pair is scored
RANSAC_ITERS = 400
# Residual tolerance as a FRACTION of depth, not metres. Monocular depth error
# scales with depth -- the lift's own DEPTH_BAND_FRAC is relative for the same
# reason -- so a fixed 0.35 m gate admits only the near field and reports a
# consensus of 3%, which is not a consensus. Everything downstream then reads as
# "inconsistent" whatever the data does.
RANSAC_TOL_FRAC = 0.04
MIN_DEPTH, MAX_DEPTH = 3.0, 40.0


def umeyama_scale(source: np.ndarray, target: np.ndarray):
    """Similarity scale mapping (3, N) source onto target, with its rotation.

    The scale is the ratio of the target's spread to the source's after both are
    centred and optimally rotated -- Umeyama's closed form. Only the scale is
    returned; the rotation and translation are the ego's motion, which is not
    what is in question here.
    """
    src_c = source - source.mean(axis=1, keepdims=True)
    tgt_c = target - target.mean(axis=1, keepdims=True)
    covariance = tgt_c @ src_c.T / source.shape[1]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0        # reflection is not a rotation
    variance = float((src_c ** 2).sum() / source.shape[1])
    if variance < 1e-9:
        return None, None
    scale = float((singular @ np.diag(correction).clip(-1, 1)).sum() / variance)
    rotation = u @ correction @ vt
    return scale, rotation


def robust_scale(source: np.ndarray, target: np.ndarray, rng):
    """RANSAC-Umeyama scale, or None. Consensus over random minimal-ish subsets."""
    n = source.shape[1]
    best_inliers, best_count = None, 0
    for _ in range(RANSAC_ITERS):
        pick = rng.choice(n, min(8, n), replace=False)
        scale, rotation = umeyama_scale(source[:, pick], target[:, pick])
        if scale is None or not (0.2 < scale < 5.0):
            continue
        aligned = scale * (rotation @ source)
        aligned += (target.mean(axis=1, keepdims=True) - aligned.mean(axis=1, keepdims=True))
        inliers = (np.linalg.norm(aligned - target, axis=0)
                   < RANSAC_TOL_FRAC * np.abs(target[2]))
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    if best_inliers is None or best_count < MIN_PAIR_POINTS // 2:
        return None, 0.0
    scale, _ = umeyama_scale(source[:, best_inliers], target[:, best_inliers])
    return scale, best_count / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--pairs", type=int, default=12, help="frame pairs to score")
    ap.add_argument("--stride", type=int, default=1,
                    help="frames between the two halves of a pair. The default compares "
                         "neighbours; a larger stride compares across the same elapsed "
                         "time as a sparser extraction, which is how a 10 Hz clip is "
                         "made comparable to a 2 Hz one (stride 5) rather than merely "
                         "easier to track.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else out_dir.parent / "frames"
    xyz_dir = out_dir / "samples-pseudodepth"
    frames = sorted(p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(frames) < 2:
        raise SystemExit(f"error: need at least two frames in {frames_dir}")

    span = args.stride
    if len(frames) <= span:
        raise SystemExit(f"error: {len(frames)} frames is too few for stride {span}")
    starts = np.linspace(0, len(frames) - 1 - span,
                         min(args.pairs, len(frames) - span)).astype(int)
    rng = np.random.default_rng(0)
    scales = []
    print(f"{'pair':<20} {'points':>7} {'inliers':>8} {'scale':>7}")
    for i in starts:
        gray0 = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        gray1 = cv2.imread(str(frames[i + span]), cv2.IMREAD_GRAYSCALE)
        xyz0_path = xyz_dir / (frames[i].stem + ".xyz.npy")
        xyz1_path = xyz_dir / (frames[i + span].stem + ".xyz.npy")
        if gray0 is None or gray1 is None or not (xyz0_path.exists() and xyz1_path.exists()):
            continue
        xyz0, xyz1 = np.load(xyz0_path), np.load(xyz1_path)

        corners = cv2.goodFeaturesToTrack(gray0, MAX_CORNERS, CORNER_QUALITY, CORNER_MIN_DIST)
        if corners is None:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            gray0, gray1, corners, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS)
        # Forward-backward: track back to frame i and keep only what lands where
        # it started. Half a second of motion produces plenty of confident-looking
        # tracks that have slid onto a different edge entirely.
        back, status_b, _ = cv2.calcOpticalFlowPyrLK(
            gray1, gray0, tracked, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS)
        round_trip = np.linalg.norm(back.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        keep = (status.ravel() == 1) & (status_b.ravel() == 1) & (round_trip < FB_TOL_PX)
        p0, p1 = corners[keep].reshape(-1, 2), tracked[keep].reshape(-1, 2)

        h, w = xyz0.shape[1:]
        c0 = np.round(p0).astype(int); c1 = np.round(p1).astype(int)
        inside = ((c0[:, 0] >= 0) & (c0[:, 0] < w) & (c0[:, 1] >= 0) & (c0[:, 1] < h) &
                  (c1[:, 0] >= 0) & (c1[:, 0] < w) & (c1[:, 1] >= 0) & (c1[:, 1] < h))
        c0, c1 = c0[inside], c1[inside]
        src = xyz0[:, c0[:, 1], c0[:, 0]]
        tgt = xyz1[:, c1[:, 1], c1[:, 0]]
        good = (np.isfinite(src).all(axis=0) & np.isfinite(tgt).all(axis=0) &
                (src[2] > MIN_DEPTH) & (src[2] < MAX_DEPTH) &
                (tgt[2] > MIN_DEPTH) & (tgt[2] < MAX_DEPTH))
        src, tgt = src[:, good], tgt[:, good]
        if src.shape[1] < MIN_PAIR_POINTS:
            print(f"{frames[i].stem}->{frames[i+1].stem:<9} {src.shape[1]:>7}   too few")
            continue
        scale, inlier_frac = robust_scale(src, tgt, rng)
        if scale is None:
            print(f"{frames[i].stem}->{frames[i+1].stem:<9} {src.shape[1]:>7}   no consensus")
            continue
        scales.append(scale)
        print(f"{frames[i].stem}->{frames[i+1].stem:<9} {src.shape[1]:>7} {inlier_frac:>7.0%} "
              f"{scale:>7.3f}")

    if len(scales) < 3:
        raise SystemExit("\nToo few scored pairs to conclude.")
    scales = np.array(scales)
    print(f"\nframe-to-frame depth scale over {len(scales)} pairs:")
    print(f"  median {np.median(scales):.4f}   p10-p90 {np.percentile(scales, 10):.3f}"
          f"-{np.percentile(scales, 90):.3f}   min-max {scales.min():.3f}-{scales.max():.3f}")
    drift = float(np.prod(scales) ** (1.0 / len(scales)))
    print(f"  geometric mean {drift:.4f} (compounded over a clip this is the trend)")
    # p90/p10, not max/min. A single pair whose LK tracks slid onto a passing
    # vehicle produces a 2x outlier and would condemn a clip whose other eleven
    # pairs agree to within a few percent -- which is exactly what max/min did on
    # the first pass over CARE_YTB. The robust spread is what the verdict reads.
    spread = float(np.percentile(scales, 90) / max(np.percentile(scales, 10), 1e-6))
    print(f"  robust spread (p90/p10) {spread:.2f}x   "
          f"outlier-driven max/min {scales.max() / max(scales.min(), 1e-6):.2f}x")
    if spread < 1.15:
        print("\n  CONSISTENT. The point maps do not change size between frames, so the\n"
              "  wander calibrate_clip.py reports is its plane fit, not the data --\n"
              "  and a single per-clip correction is the right shape of fix.")
    else:
        print(f"\n  INCONSISTENT: {spread:.2f}x between the tightest and loosest pair.\n"
              "  The depth field really does change size frame to frame, so there is no\n"
              "  one camera to correct to.")


if __name__ == "__main__":
    main()
