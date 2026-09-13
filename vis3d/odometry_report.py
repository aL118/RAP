#!/usr/bin/env python3
"""What every clip's odometry actually produced, and where it stopped.

    python odometry_report.py --data ../data/test --run 1

run_odometry.sh calls this as its last stage. It exists because a batch log is
not a result: the interesting question after a sweep is not "did it print an
error" but "which clips have a usable trajectory, which stage stopped the rest,
and on the clips where the answer can be checked, how wrong is it".

The accuracy column is deliberately DISTANCE, not mean speed. Mean speed is
taken over matched frames only, so a clip that matches 9% of its frames can
report a flattering average while having integrated almost no motion --
four_way reads -1.8 km/h by mean speed and 2% of true distance. Distance counts
the frames that produced nothing, which is what a trajectory consumer feels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

STAGES = [
    ("lane_masks", "lane_masks", "dir"),
    ("intrinsics", "camera_intrinsics.json", "file"),
    ("depth", "samples-pseudodepth", "dir"),
    ("poses", "ego_poses.txt", "file"),
    ("traj.png", "ego_trajectory.png", "file"),
    ("plane", "road_plane.json", "file"),
    ("dashes", "dashes.json", "file"),
    ("speed", "dash_speed.json", "file"),
    ("dash.png", "dash_odometry_vs_ground_truth.png", "file"),
    ("road", "road_speed.json", "file"),
    ("chosen", "ego_speed.json", "file"),
]
SPEED_SOURCES = (("dash", "dash_speed.json"), ("road", "road_speed.json"),
                 ("chosen", "ego_speed.json"))


def truth_curve(entry):
    n = entry["frames"]
    samples = np.arange(0, n, entry["sample_step"])[:len(entry["speed_kmh"])]
    values = np.array([np.nan if v is None else v for v in entry["speed_kmh"]])
    good = np.isfinite(values)
    return np.interp(np.arange(n), samples[good], values[good], right=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--clips", default=None, help="space-separated; default every clip dir")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    data = Path(args.data).resolve()
    clips = args.clips.split() if args.clips else sorted(
        p.name for p in data.iterdir() if (p / "frames").is_dir())
    truth = {}
    if args.gt and Path(args.gt).exists():
        truth = json.loads(Path(args.gt).read_text())["clips"]

    rows, report = [], {}
    for clip in clips:
        out = data / clip / args.run
        present = {}
        for label, name, kind in STAGES:
            target = out / name
            present[label] = (target.is_dir() and any(target.iterdir())) if kind == "dir" \
                else target.is_file()
        entry = {"stages": present}

        # where it stopped: the first missing stage after the last present one
        order = [label for label, _, _ in STAGES]
        done = [label for label in order if present[label]]
        entry["stopped_after"] = done[-1] if done else "nothing"

        plane = out / "road_plane.json"
        if plane.exists():
            try:
                entry["plane_quality"] = json.loads(plane.read_text()).get("quality", "legacy")
            except Exception:
                entry["plane_quality"] = "?"

        accuracy = {}
        for source, name in SPEED_SOURCES:
            if clip not in truth or not (out / name).exists():
                continue
            speed = json.loads((out / name).read_text())
            estimate = np.array([np.nan if v is None else v for v in speed["speed_kmh"]])
            reference = truth_curve(truth[clip])
            m = min(len(estimate), len(reference))
            estimate, reference = estimate[:m], reference[:m]
            matched = np.isfinite(estimate)
            # the speed file's own rate: a clip need not be at the --hz default
            hz = float(speed.get("hz") or args.hz)
            est_m = float(np.nansum(np.where(matched, estimate, 0.0)) / 3.6 / hz)
            true_m = float(np.sum(reference) / 3.6 / hz)
            accuracy[source] = {"matched_fraction": round(float(matched.mean()), 3),
                                "distance_m": round(est_m, 1), "truth_m": round(true_m, 1),
                                "distance_ratio": round(est_m / true_m, 3) if true_m > 1 else None}
        # "accuracy" keeps meaning dash odometry, as it did before road_speed existed
        if "dash" in accuracy:
            entry["accuracy"] = accuracy["dash"]
        if "road" in accuracy:
            entry["road_accuracy"] = accuracy["road"]
        report[clip] = entry
        rows.append((clip, present, entry["stopped_after"], accuracy,
                     entry.get("plane_quality", "")))

    width = max(max(len(c) for c in clips) + 2, 16)
    header = (f"{'clip':<{width}}" + "".join(f"{label:>10}" for label, _, _ in STAGES)
              + f"{'plane qual':>16}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for clip, present, stopped, accuracy, quality in rows:
        marks = "".join(f"{('  yes' if present[label] else '   --'):>10}"
                        for label, _, _ in STAGES)
        print(f"{clip:<{width}}{marks}{quality:>16}")
    complete = sum(1 for _, p, _, _, _ in rows if p["dashes"])
    print(f"\n{len(rows)} clips; {sum(1 for _, p, _, _, _ in rows if p['intrinsics'])} with intrinsics, "
          f"{sum(1 for _, p, _, _, _ in rows if p['traj.png'])} with a baseline trajectory plot, "
          f"{complete} with dashes detected, "
          f"{sum(1 for _, p, _, _, _ in rows if p['road'])} with a road-surface speed")
    by_quality = {}
    for _, _, _, _, q in rows:
        if q:
            by_quality[q] = by_quality.get(q, 0) + 1
    if by_quality:
        print("  plane quality: " + ", ".join(f"{v} {k}" for k, v in sorted(by_quality.items())))
        if set(by_quality) - {"measured"}:
            print("  anything but 'measured' is a fallback, not a calibration -- its speeds\n"
                  "  inherit a plane nobody verified.")

    for source, _ in SPEED_SOURCES:
        checked = [(c, a[source]) for c, _, _, a, _ in rows
                   if source in a and a[source]["distance_ratio"] is not None]
        if not checked:
            continue
        print(f"\n{source} odometry against burned-in ground truth (distance, not mean speed):")
        print(f"  {'clip':<16}{'matched':>9}{'est m':>8}{'truth m':>9}{'ratio':>8}")
        for clip, a in sorted(checked, key=lambda r: -r[1]["distance_ratio"]):
            print(f"  {clip:<16}{a['matched_fraction']*100:>8.0f}%{a['distance_m']:>8.0f}"
                  f"{a['truth_m']:>9.0f}{a['distance_ratio']*100:>7.0f}%")
        ratios = [a["distance_ratio"] for _, a in checked]
        print(f"  {'median':<16}{'':>9}{'':>8}{'':>9}{np.median(ratios)*100:>7.0f}%")
    print("\n  A ratio well under 100% means frames that produced no match contributed no\n"
          "  distance. Do not read these as training-ready labels.")

    destination = data / "odometry_report.json"
    destination.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
