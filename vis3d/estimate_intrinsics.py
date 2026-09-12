"""
Estimates a clip's camera intrinsics with WildCamera, writing the one K that
infer_unidepth_on_frames.py --intrinsics conditions UniDepth on.

Why this stage exists: UniDepth predicts its own camera as a side effect of
predicting depth, and on wide-angle dashcam footage that estimate is often
roughly twice the true focal length -- a ~42 deg horizontal FOV read off a lens
that is far wider. Because it conditions its own depth on that camera, a long
fx does not merely mislabel the result; it collapses it. On the CARE clips where
it happens the whole point map becomes a frontoparallel slab: changelane frame 0
spans 21.9-38.9 m, the car's own hood and the horizon at the same range, no
ground-plane gradient at all. Odometry has nothing left to triangulate against,
which is how five freeway clips ended up with an implied ego speed of 0.1-0.2
m/s (changelane's own burned-in GPS says it covered 265 m).

WildCamera is an independent second opinion, and that independence is the whole
point: there is no calibration for any of this footage, so the only way to know
UniDepth's camera is wrong is for a differently-trained model to disagree with
it. Measured over 19 CARE clips, the two agree within ~12% on every clip whose
depth is healthy (blocker 1.12x, uturn 1.03x, barrier_whip 0.93x, deercross
0.92x) and UniDepth runs 1.6-2.3x long on every clip whose depth has collapsed.
The ratio this stage prints is therefore a usable screen on its own, and a
cheaper one than looking at the depth: it needs no GPU and one frame's forward
pass.

One K for the whole clip, not one per frame. It is the same physical lens
throughout, and UniDepth's per-frame estimate wandering 2148-2662 within a
single clip is itself an error source -- it makes the metric scale of the point
maps drift frame to frame, which is exactly what the box tracker and the
odometry are least able to absorb.

Runs in the 'processor' env, not 'vis3d': WildCamera's decoder imports
mmcv.cnn, which vis3d does not have. That is the same arrangement every other
borrowed model in this pipeline has -- see estimate_ego_motion.py on OpenVO --
and it is why this is its own stage rather than a few lines inside stage 2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
DEFAULT_REPO = Path("/fs/nexus-projects/sim2real/aliu/OpenVO/processor/WildCamera")
DEFAULT_WEIGHTS = Path(
    "/fs/nexus-projects/sim2real/aliu/OpenVO/weights/WildCamera/wild_camera_all.pth")

# Frames to sample. The estimate is stable enough that a handful would do (the
# per-frame spread on hydroplaning is 37 px on a 1244 px focal, 3%); the point
# of taking more is the median, which costs a second a frame on CPU and makes
# the result immune to a single blown-out or motion-blurred frame.
DEFAULT_SAMPLES = 16

# Beyond this spread across sampled frames the estimate is not trustworthy
# enough to condition depth on -- WildCamera is usually far tighter than this,
# and a wide spread means the clip is giving it nothing to work with.
FOCAL_SPREAD_WARN = 0.12

# UniDepth-vs-WildCamera focal ratio beyond which UniDepth's camera is treated
# as the broken one. Set between the two clusters measured across CARE: healthy
# clips land at 0.92-1.12x, collapsed ones at 1.6-2.3x.
RATIO_WARN = 1.25


def load_model(repo: Path, weights: Path):
    """WildCamera's NEWCRFIF with its released weights, on CPU or GPU.

    The repo is put on sys.path rather than imported as an installed package:
    it is installed editable into the 'processor' env, but this keeps the stage
    runnable from a checkout that has not been pip-installed.
    """
    import torch

    sys.path.insert(0, str(repo))
    from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF  # noqa: E402

    model = NEWCRFIF(version="large07", pretrained=None)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()


def estimate(model, frame_paths, four_dof: bool) -> np.ndarray:
    """Per-frame 3x3 intrinsics, stacked (N, 3, 3).

    wtassumption=True is WildCamera's 1-DoF solve: square pixels, principal
    point at the image centre, focal length the only unknown. It is the default
    here because the 4-DoF solve buys little and costs stability -- on
    hydroplaning it agrees on fx to 1.3% but puts the principal point 37 px
    right of centre with a per-frame spread of 13 px in cy, and a K that jitters
    frame to frame is worse for this pipeline than one that is slightly off in a
    fixed direction.
    """
    return np.stack([
        model.inference(Image.open(path).convert("RGB"), wtassumption=not four_dof)[0]
        for path in tqdm(frame_paths, desc="intrinsics")
    ])


def _unidepth_focal(xyz_dir: Path, frame_paths) -> float:
    """Median fx UniDepth predicted for the same frames, or nan if stage 2 has
    not run yet. Only used for the divergence warning."""
    focals = []
    for path in frame_paths:
        k_path = xyz_dir / f"{path.stem}.K.npy"
        if k_path.exists():
            focals.append(np.load(k_path).reshape(3, 3)[0, 0])
    return float(np.median(focals)) if focals else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate one camera K per clip, for infer_unidepth_on_frames.py --intrinsics.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory of frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Run directory; camera_intrinsics.json is written here.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Frames to sample across the clip and take the median of.")
    parser.add_argument("--four_dof", action="store_true",
                        help="Solve fx, fy, cx, cy instead of focal length alone. Noisier; "
                             "use only when the footage is known to be cropped off-centre.")
    parser.add_argument("--repo", type=str, default=str(DEFAULT_REPO),
                        help="WildCamera checkout.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="WildCamera checkpoint.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    frame_paths = sorted(p for p in frames_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert frame_paths, f"No frame images found in {frames_dir}"
    sampled = [frame_paths[i] for i in
               np.unique(np.linspace(0, len(frame_paths) - 1, args.samples).astype(int))]

    width, height = Image.open(frame_paths[0]).size
    model = load_model(Path(args.repo), Path(args.weights))
    intrinsics = estimate(model, sampled, args.four_dof)

    # Median rather than mean: one frame of the clip being a wash of rain or
    # headlight glare should move this not at all.
    camera = np.median(intrinsics, axis=0)
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    focals = intrinsics[:, 0, 0]
    spread = float(focals.std() / focals.mean())
    fov = float(2 * np.degrees(np.arctan(0.5 * width / fx)))

    destination = output_dir / "camera_intrinsics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(destination, "w") as file:
        json.dump({
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height,
            "hfov_deg": fov,
            "source": "wildcamera-4dof" if args.four_dof else "wildcamera-1dof",
            "frames_sampled": len(sampled),
            "focal_spread": spread,
        }, file, indent=2)

    print(f"fx {fx:.1f}  fy {fy:.1f}  cx {cx:.1f}  cy {cy:.1f}   "
          f"({fov:.1f} deg horizontal FOV, {len(sampled)} frames, spread {spread:.1%})")
    if spread > FOCAL_SPREAD_WARN:
        print(f"Warning: focal length varies {spread:.1%} across frames on what is one fixed "
              "lens. Treat this K as weak evidence and check the depth it produces.")

    predicted = _unidepth_focal(output_dir / "samples-pseudodepth", sampled)
    if np.isfinite(predicted):
        ratio = predicted / fx
        print(f"UniDepth's own fx on these frames: {predicted:.1f} ({ratio:.2f}x this estimate)")
        if ratio > RATIO_WARN or ratio < 1 / RATIO_WARN:
            print("  The two cameras disagree. The existing point maps were built on UniDepth's, "
                  "so re-run stage 2 with --intrinsics to rebuild them on this one.")
        else:
            print("  The two cameras agree; this clip's existing point maps are probably fine.")

    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
