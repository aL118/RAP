"""
Runs UniDepth over a plain directory of video frames (e.g. nightcrash dashcam
frames), saving each frame's dense per-pixel point map to disk.

Adapted from infer_unidepth_on_nuscenes.py, but much simpler: there's a single
uncalibrated camera and no ego pose / multi-camera loop to drive, so this just
walks the frame files directly. Runs in the 'vis3d' env, whose torch 2.2.0 /
xformers 0.0.24 pins exist for this stage: UniDepth's decoder uses xformers'
NystromAttention, which xformers deleted after 0.0.24 (see
scripts/vis3d/requirements.txt).

--intrinsics is the difference between a point map that is merely imprecise and
one that is unusable. Left to itself UniDepth predicts the camera along with the
depth and conditions one on the other, so a focal length that comes out ~2x too
long does not just mislabel the result -- it flattens it. On the CARE clips
where that happens the entire point map becomes a frontoparallel slab
(changelane frame 0: 21.9-38.9 m, the hood and the horizon at the same range,
no ground-plane gradient), and the ego motion estimated from it collapses to
0.1-0.2 m/s on footage whose own burned-in GPS says 265 m. Give it a K from
estimate_intrinsics.py and it conditions on that instead.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from unidepth.models import UniDepthV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A healthy point map of a road scene spans a large depth range: something close
# (the ego's own hood, the car ahead) and something far. Measured across CARE,
# clips whose depth is usable have a 99th/1st percentile ratio of 10-30 and a
# nearest point 1-6 m away; every clip whose odometry had collapsed was under
# 2.5 with nothing nearer than 10 m. Reported at the end of the run because it
# is the cheapest way to tell a good stage 2 from a bad one, and because a bad
# one is otherwise invisible until the trajectory comes out wrong two stages later.
HEALTHY_DEPTH_RATIO = 2.5


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def frame_images(frames_dir: Path):
    """The clip's frame files, in order."""
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_intrinsics(spec: str) -> np.ndarray:
    """A 3x3 K from either estimate_intrinsics.py's json or a literal
    "fx,fy,cx,cy"."""
    path = Path(spec)
    if path.exists():
        with open(path) as file:
            camera = json.load(file)
        fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    else:
        try:
            fx, fy, cx, cy = (float(v) for v in spec.split(","))
        except ValueError:
            raise SystemExit(
                f"--intrinsics {spec!r} is neither a file nor an 'fx,fy,cx,cy' quadruple")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def infer_image(image_path: Path, model: UniDepthV1, intrinsics: np.ndarray = None):
    """Returns (point map, intrinsics): the dense camera-frame point map, shape
    (3, H, W) in x-right/y-down/z-forward convention, and the 3x3 pinhole
    intrinsics the point map is expressed in.

    With no `intrinsics`, that is whatever UniDepth predicted for the frame --
    there is no calibration for this footage, so its estimate is the only camera
    we have. With one, UniDepth conditions its decoder on it
    (`test_fixed_camera=True`) and back-projects through it, so the camera
    returned is the one passed in.

    Note the returned K is deliberately *not* `predictions["intrinsics"]` when a
    camera was supplied: UniDepth still reports its own guess there even when it
    was told what the camera is, and saving that would leave every .K.npy
    disagreeing with the point map beside it -- which lift_frames_to_3d.py
    checks for and silently works around, so nothing would fail, it would just
    quietly use a different camera than the one this stage was run with.
    """
    rgb = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1)
    camera = None if intrinsics is None else torch.from_numpy(intrinsics)
    with torch.no_grad():
        predictions = model.infer(rgb.to(DEVICE), camera)
    if intrinsics is None:
        return (predictions["points"].squeeze(0).cpu().numpy(),
                predictions["intrinsics"].squeeze(0).cpu().numpy())
    return _backproject(predictions["depth"], intrinsics).squeeze(0).cpu().numpy(), intrinsics


def _backproject(depth: torch.Tensor, intrinsics: np.ndarray) -> torch.Tensor:
    """Rebuilds the point map from UniDepth's depth and the camera we asked for.

    Works around a bug in UniDepthV1.infer: `_preprocess` scales a supplied K up
    to the network's own input resolution, `_postprocess` scales only the
    *predicted* K back down, and the final back-projection then runs the
    network-resolution K against the full-resolution image. On a 1920-wide frame
    UniDepth works at 616, so the point map comes out built on fx/3.117 -- x and
    y three times too large, while z, which is the network's depth output
    unaltered, stays correct. Passing intrinsics is the only way to hit it,
    which is presumably why it survives upstream.

    Rebuilding from `depth` rather than rescaling `points` because that is the
    one quantity the bug does not touch: this is the same spherical-to-euclidean
    step infer() ends with, just handed the K at the resolution it belongs to.
    """
    from unidepth.utils.geometric import generate_rays, spherical_zbuffer_to_euclidean

    height, width = depth.shape[-2:]
    camera = torch.from_numpy(intrinsics).unsqueeze(0).to(depth.device)
    angles = generate_rays(camera, (height, width), noisy=False)[-1]
    angles = angles.reshape(1, height, width, 2).permute(0, 3, 1, 2)
    spherical = torch.cat((angles, depth), dim=1).permute(0, 2, 3, 1)
    return spherical_zbuffer_to_euclidean(spherical).permute(0, 3, 1, 2)


def predict_frames(model: UniDepthV1, frames_dir: Path, output_dir: Path,
                   intrinsics: np.ndarray = None) -> None:
    frame_paths = frame_images(frames_dir)
    assert frame_paths, f"No frame images found in {frames_dir}"

    xyz_dir = output_dir / "samples-pseudodepth"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    ranges, nearest = [], []
    for frame_path in tqdm(frame_paths):
        xyz_path = xyz_dir / frame_path.with_suffix(".xyz.npy").name
        k_path = xyz_dir / frame_path.with_suffix(".K.npy").name
        if xyz_path.exists() and k_path.exists() and _matches(k_path, intrinsics):
            xyz = np.load(xyz_path)
        else:
            xyz, camera = infer_image(frame_path, model, intrinsics)
            np.save(xyz_path, xyz)
            np.save(k_path, camera)
        low, high = np.percentile(xyz[2], [1, 99])
        ranges.append(high / max(low, 1e-6))
        nearest.append(float(xyz[2].min()))

    ratio, near = float(np.median(ranges)), float(np.median(nearest))
    print(f"Depth spread: 99th/1st percentile {ratio:.1f}x, nearest point {near:.1f} m "
          f"(median over {len(frame_paths)} frames)")
    if ratio < HEALTHY_DEPTH_RATIO:
        print("Warning: this point map is nearly frontoparallel -- the whole scene sits in a "
              "thin slab at one range, with no ground plane receding from the camera. Boxes "
              "lifted from it will be wrong and ego motion estimated from it will collapse "
              "toward zero. Run estimate_intrinsics.py and re-run this stage with "
              "--intrinsics; the cause is usually a predicted focal length about twice the "
              "true one.")


def _matches(k_path: Path, intrinsics: np.ndarray) -> bool:
    """Whether an already-written point map was built on the camera being asked
    for now.

    Without this, re-running the stage with a corrected --intrinsics over a
    directory that already holds point maps would skip every frame and leave the
    old, wrong ones in place -- the resume check would report success having
    changed nothing.
    """
    if intrinsics is None:
        return True
    return np.allclose(np.load(k_path).reshape(3, 3), intrinsics, rtol=1e-3, atol=1e-3)


def main():
    parser = argparse.ArgumentParser(description="Run UniDepth over a directory of video frames.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory containing frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write samples-pseudodepth/<frame>.xyz.npy into.")
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="Camera to condition on: estimate_intrinsics.py's "
                             "camera_intrinsics.json, or a literal 'fx,fy,cx,cy'. Omit to let "
                             "UniDepth predict its own, which on wide-angle dashcam footage is "
                             "often ~2x too long and collapses the depth.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)

    intrinsics = None
    if args.intrinsics is not None:
        intrinsics = load_intrinsics(args.intrinsics)
        # Off the frame's own width rather than 2*cx, which is the same thing
        # only while the principal point is centred (--four_dof moves it).
        width = Image.open(frame_images(frames_dir)[0]).size[0]
        fov = 2 * np.degrees(np.arctan(0.5 * width / intrinsics[0, 0]))
        print(f"Conditioning on fx {intrinsics[0, 0]:.1f}, fy {intrinsics[1, 1]:.1f}, "
              f"cx {intrinsics[0, 2]:.1f}, cy {intrinsics[1, 2]:.1f} ({fov:.1f} deg horizontal FOV)")

    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    model = model.to(torch.device(DEVICE)).eval()

    predict_frames(model, frames_dir, output_dir, intrinsics)


if __name__ == "__main__":
    main()
