#!/usr/bin/env python3
"""
Plots the ego trajectory estimate_ego_motion.py produced, as a sanity check on
it before anything trains on it.

Separate from estimate_ego_motion.py because that runs in whichever env its
method needs -- 'vis3d' for pointcloud, 'openvo' for openvo -- and vis3d has no
matplotlib. This reads only the pose file, so it runs anywhere matplotlib does
and can be pointed at a trajectory estimated weeks ago.

The three panels are chosen to catch the two ways a VO run has failed on this
footage, neither of which a single summary number exposes:

  - the path, in the ego frame. snowcrash's trajectory sat still for 333 of its
    406 frames; that is obvious here and invisible in an average speed.
  - speed over time, which catches a scale error (OpenVO's released checkpoints
    were trained on LiDAR depth and under-predict dashcam motion by ~20x) and
    the jumps a failed frame pair leaves behind.
  - cumulative heading. changelane's pointcloud run looked healthy on speed --
    a median 0.96 m per frame, ~35 km/h, no outliers -- while accumulating a
    median 1.6 deg of yaw *per frame*, so its 601 m of path curled into 48 m of
    net displacement. Straightness (net / path) is printed on the first panel
    for the same reason: a motorway clip should be near 1.0.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt

# Camera (x right, y down, z forward) -> ego (x forward, y left, z up). The
# pose file is KITTI-format camera-to-world; plotting in the ego frame means
# the panels show what navsim's trajectory label actually is, not a rotated
# version of it. Same remap as export_navsim_logs.load_ego_poses.
CAMERA_TO_EGO_AXES = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])


def load_poses(poses_path: Path) -> np.ndarray:
    """(N, 4, 4) ego-to-world transforms from a KITTI-format pose file."""
    rows = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = CAMERA_TO_EGO_AXES @ rows[:, :3, :3] @ CAMERA_TO_EGO_AXES.T
    poses[:, :3, 3] = rows[:, :3, 3] @ CAMERA_TO_EGO_AXES.T
    return poses


def summarize(poses: np.ndarray, hz: float) -> dict:
    """The numbers printed on the plot and to stdout."""
    position = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    path = float(steps.sum())
    net = float(np.linalg.norm(position[-1] - position[0]))

    # Heading from the forward axis, unwrapped so a full turn accumulates
    # instead of wrapping back to zero.
    forward = poses[:, :3, 0]
    heading = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))

    return {
        "frames": len(poses),
        "path_m": path,
        "net_m": net,
        "straightness": net / path if path > 1e-9 else 0.0,
        "speed_kmh": steps * hz * 3.6,
        "median_kmh": float(np.median(steps) * hz * 3.6),
        "mean_kmh": float(steps.mean() * hz * 3.6),
        "stalled": int((steps < 1e-9).sum()),
        "heading_deg": np.degrees(heading - heading[0]),
        "position": position,
    }


def plot(stats: dict, hz: float, title: str, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    figure.suptitle(title, fontsize=13)

    x, y = stats["position"][:, 0], stats["position"][:, 1]
    ax = axes[0]
    # Coloured by frame index so the direction of travel, and any doubling
    # back, is readable without arrows.
    ax.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=6, zorder=2)
    ax.plot(x, y, lw=0.8, color="0.6", zorder=1)
    ax.plot(x[0], y[0], "o", color="tab:green", ms=9, label="start", zorder=3)
    ax.plot(x[-1], y[-1], "s", color="tab:red", ms=8, label="end", zorder=3)
    ax.set_aspect("equal", adjustable="datalim")   # a turn must look like a turn
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(f"path {stats['path_m']:.0f} m   net {stats['net_m']:.0f} m   "
                 f"straightness {stats['straightness']:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    time = np.arange(len(stats["speed_kmh"])) / hz
    ax = axes[1]
    ax.plot(time, stats["speed_kmh"], lw=1.0)
    ax.axhline(stats["median_kmh"], color="tab:orange", ls="--", lw=1,
               label=f"median {stats['median_kmh']:.0f} km/h")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (km/h)")
    ax.set_title(f"speed   mean {stats['mean_kmh']:.0f} km/h   "
                 f"stalled frames {stats['stalled']}/{stats['frames'] - 1}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    ax.plot(np.arange(stats["frames"]) / hz, stats["heading_deg"], lw=1.0)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative heading (deg)")
    ax.set_title(f"heading   total {stats['heading_deg'][-1]:+.0f} deg")
    ax.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Run directory holding ego_poses.txt; the plot is written here.")
    parser.add_argument("--poses", type=str, default=None,
                        help="Pose file (default: <output_dir>/ego_poses.txt).")
    parser.add_argument("--out", type=str, default=None,
                        help="Plot path (default: <output_dir>/ego_trajectory.png).")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Rate the frames the poses were estimated from were extracted at.")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title (default: the run directory's name).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    poses_path = Path(args.poses) if args.poses else output_dir / "ego_poses.txt"
    if not poses_path.exists():
        raise FileNotFoundError(f"{poses_path} does not exist; run estimate_ego_motion.py first.")
    destination = Path(args.out) if args.out else output_dir / "ego_trajectory.png"

    poses = load_poses(poses_path)
    stats = summarize(poses, args.hz)
    title = args.title or f"{output_dir.parent.name}/{output_dir.name}  ({poses_path.name})"
    plot(stats, args.hz, title, destination)

    print(f"{stats['frames']} poses: path {stats['path_m']:.0f} m, net {stats['net_m']:.0f} m, "
          f"straightness {stats['straightness']:.2f}, median {stats['median_kmh']:.0f} km/h, "
          f"{stats['stalled']} stalled frames, heading {stats['heading_deg'][-1]:+.0f} deg")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
