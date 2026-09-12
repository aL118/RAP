#!/usr/bin/env python3
"""Dash odometry against the burned-in OSD ground truth: speed, distance, path.

    conda activate vis3d
    python detect_dashes.py  --clip ../data/CARE_YTB/changelane \
        --intrinsics ../data/test/changelane/1/camera_intrinsics.json \
        --pitch_deg 0.70 --height 1.33
    python plot_dash_odometry.py --clip ../data/CARE_YTB/changelane \
        --intrinsics ../data/test/changelane/1/camera_intrinsics.json \
        --out ../data/test/changelane/1/dash_odometry_vs_ground_truth.png

The heading pass needs the 'vis3d' env (opencv); the plotting needs 'rap'
(matplotlib), so the script runs the heading pass first and caches it to
dash_heading.npy -- re-plotting afterwards works in either env.

WHAT THE TRAJECTORY PANEL IS, AND WHAT IT IS NOT

dash_odometry.py measures *speed* only. A path needs a heading too, and that
comes from the essential matrix -- the half of ground_plane_odometry.py that
works (see its STATUS note: rotation and translation direction are sound, only
the metric scale was missing). So the plotted path is dash-odometry scale on
essential-matrix heading, which is the first complete trajectory estimate on
this footage that has a measured scale rather than a depth network's guess.

The ground-truth path is drawn as a straight line of the OSD-integrated length.
That is honest for changelane and only for clips like it: the GPS endpoints are
265 m apart and the integrated speed is 277 m, i.e. the route is straight to
within 4%, so a straight line IS the truth here. On a turning clip it would not
be, and the panel says so rather than implying a real GPS track.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HZ = 10.0


def heading_from_essential(frames_dir, run_dir, intrinsics, cache):
    """Cumulative yaw per frame from the essential matrix, cached to disk."""
    if Path(cache).exists():
        return np.load(cache)
    import cv2, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from point_cloud_odometry import _track
    from estimate_ego_motion import _static_mask, _load_object_masks

    names = sorted(p.name for p in Path(frames_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".png"})
    masks = _load_object_masks(Path(run_dir))
    previous = cv2.imread(str(Path(frames_dir) / names[0]), cv2.IMREAD_GRAYSCALE)
    shape = previous.shape
    yaw = [0.0]
    for name in names[1:]:
        gray = cv2.imread(str(Path(frames_dir) / name), cv2.IMREAD_GRAYSCALE)
        static = _static_mask(masks, name, shape).astype(bool)
        step = 0.0
        tracked = _track(previous, gray, static)
        if tracked is not None and len(tracked[0]) >= 40:
            essential, mask = cv2.findEssentialMat(
                tracked[0], tracked[1], intrinsics, method=cv2.RANSAC,
                prob=0.999, threshold=1.5)
            if essential is not None and essential.shape == (3, 3) and int(mask.sum()) >= 40:
                _, rotation, _, _, _ = cv2.recoverPose(
                    essential, tracked[0], tracked[1], intrinsics,
                    mask=mask.copy(), distanceThresh=1e9)
                # yaw about the camera's down axis (y)
                step = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        yaw.append(yaw[-1] + step)
        previous = gray
    yaw = np.array(yaw)
    np.save(cache, yaw)
    return yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--gt", default=None, help="osd_ground_truth.json")
    parser.add_argument("--name", default=None, help="clip key in the ground-truth file")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_heading", action="store_true",
                        help="skip the essential-matrix pass and the path panel")
    args = parser.parse_args()

    clip = Path(args.clip).resolve()
    run = clip / args.run
    name = args.name or clip.name
    root = Path(__file__).resolve().parent.parent

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dash_odometry as DO

    gt_path = Path(args.gt) if args.gt else root / "data/test/osd_ground_truth.json"
    truth = json.loads(gt_path.read_text())["clips"].get(name)
    if truth is None:
        raise SystemExit(f"{name} has no burned-in ground truth in {gt_path}")

    info = clip / "info.json"
    country = json.loads(info.read_text()).get("country") if info.exists() else None
    cycle = DO.DASH_CYCLE_M.get(country, DO.DASH_CYCLE_M[None])
    kmh, advance, correction = DO.speed(run / "dashes.json", cycle, False, HZ)
    kmh_sym, _, _ = DO.speed(run / "dashes.json", cycle, True, HZ)

    n = truth["frames"]
    samples = np.arange(0, n, truth["sample_step"])[:len(truth["speed_kmh"])]
    values = np.array([np.nan if v is None else v for v in truth["speed_kmh"]])
    good = np.isfinite(values)
    gt = np.interp(np.arange(n), samples[good], values[good], right=0.0)

    yaw = None
    if not args.no_heading:
        kpath = Path(args.intrinsics) if args.intrinsics else run / "camera_intrinsics.json"
        k = json.loads(Path(kpath).read_text())
        intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
        yaw = heading_from_essential(clip / "frames", run, intrinsics, run / "dash_heading.npy")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(v, k=9):
        return np.array([np.nanmedian(v[max(0, i-k//2):i+k//2+1]) for i in range(len(v))])

    m = min(n, len(kmh))
    t = np.arange(m) / HZ
    ncols = 4 if yaw is not None else 3
    fig, ax = plt.subplots(1, ncols, figsize=(5.0*ncols, 5.2))

    ax[0].plot(t, gt[:m], "k", lw=3, label="ground truth (OSD)", zorder=5)
    ax[0].plot(t, kmh[:m], color="tab:blue", lw=.7, alpha=.35, label="dash odometry, per frame")
    ax[0].plot(t, smooth(kmh[:m]), color="tab:blue", lw=2.2, label="dash odometry, 9-frame median")
    still = gt[:m] < 1.0
    if still.any():
        ax[0].axvspan(t[still][0], t[-1], color="0.88", zorder=0)
    ax[0].set(xlabel="time (s)", ylabel="speed (km/h)", title=f"{name}: speed vs ground truth")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    for label, v, c, lw in (("ground truth", gt[:m], "k", 3),
                            ("dash odometry", kmh[:m], "tab:blue", 2),
                            ("symmetric control", kmh_sym[:m], "tab:cyan", 1.5)):
        d = np.cumsum(np.nan_to_num(v)/3.6/HZ)
        ax[1].plot(t, d, c, lw=lw, label=f"{label}  ({d[-1]:.0f} m)")
    ax[1].set(xlabel="time (s)", ylabel="distance (m)", title="cumulative distance")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    if yaw is not None:
        step = np.nan_to_num(kmh[:m])/3.6/HZ
        head = yaw[:m]
        xy = np.cumsum(np.c_[step*np.cos(head), step*np.sin(head)], axis=0)
        gt_len = np.cumsum(gt[:m]/3.6/HZ)[-1]
        ax[2].plot([0, gt_len], [0, 0], "k", lw=3, label=f"ground truth ({gt_len:.0f} m, straight)")
        ax[2].plot(xy[:,0], xy[:,1], color="tab:blue", lw=2,
                   label=f"dash speed + E-matrix heading ({np.linalg.norm(xy[-1]):.0f} m net)")
        ax[2].scatter([0], [0], c="tab:green", s=70, zorder=5, label="start")
        ax[2].scatter([xy[-1,0]], [xy[-1,1]], c="tab:red", marker="s", s=70, zorder=5, label="end")
        ax[2].set(xlabel="x forward (m)", ylabel="y left (m)", title="path in the ego frame")
        ax[2].axis("equal"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    a = ax[-1]
    idx = np.flatnonzero(gt[:m] < 1.0)
    rows = [("ground truth", 0.0, "k"),
            ("dash odometry", float(np.nanmedian(kmh[:m][idx])) if len(idx) else np.nan, "tab:blue"),
            ("symmetric control", float(np.nanmedian(kmh_sym[:m][idx])) if len(idx) else np.nan, "tab:cyan")]
    a.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], alpha=.85)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=9)
    for i, r in enumerate(rows):
        if np.isfinite(r[1]): a.text(r[1], i, f"  {r[1]:.2f}", va="center", fontsize=9)
    a.set(xlabel="median speed (km/h)",
          title=f"stationary control ({len(idx)} frames)\n(truth is exactly 0)")
    a.grid(alpha=.3, axis="x"); a.invert_yaxis()

    fig.suptitle(f"{name}: lane-dash odometry vs burned-in OSD ground truth "
                 f"(dash cycle {cycle} m, plane correction {correction:.2f}x)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out) if args.out else run / "dash_odometry_vs_ground_truth.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    mv = gt[:m] > 20
    if mv.any():
        print(f"moving median {np.nanmedian(kmh[:m][mv]):.1f} km/h vs GT {np.median(gt[:m][mv]):.1f}")


if __name__ == "__main__":
    main()
