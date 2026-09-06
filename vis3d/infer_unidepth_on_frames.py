"""
Runs UniDepth over a plain directory of video frames (e.g. nightcrash dashcam
frames), saving each frame's dense per-pixel point map to disk.

Adapted from infer_unidepth_on_nuscenes.py, but much simpler: there's a single
uncalibrated camera and no ego pose / multi-camera loop to drive, so this just
walks the frame files directly. Runs in the 'vis3d' env, whose torch 2.2.0 /
xformers 0.0.24 pins exist for this stage: UniDepth's decoder uses xformers'
NystromAttention, which xformers deleted after 0.0.24 (see
scripts/vis3d/requirements.txt).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from unidepth.models import UniDepthV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def infer_image(image_path: Path, model: UniDepthV1):
    """Returns (point map, intrinsics): the dense camera-frame point map, shape
    (3, H, W) in x-right/y-down/z-forward convention, and the 3x3 pinhole
    intrinsics UniDepth predicted for the frame (there is no calibration for
    this footage, so its estimate is the only camera we have -- and it is the
    camera the point map is expressed in, so lift_frames_to_3d.py and
    visualization/raster_frames.py must both use it or the boxes land nowhere
    near the objects they came from)."""
    rgb = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1)
    with torch.no_grad():
        predictions = model.infer(rgb.to(DEVICE))
    return (predictions["points"].squeeze(0).cpu().numpy(),
            predictions["intrinsics"].squeeze(0).cpu().numpy())


def predict_frames(model: UniDepthV1, frames_dir: Path, output_dir: Path) -> None:
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    assert frame_paths, f"No frame images found in {frames_dir}"

    xyz_dir = output_dir / "samples-pseudodepth"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in tqdm(frame_paths):
        xyz_path = xyz_dir / frame_path.with_suffix(".xyz.npy").name
        k_path = xyz_dir / frame_path.with_suffix(".K.npy").name
        if xyz_path.exists() and k_path.exists():
            continue
        xyz, intrinsics = infer_image(frame_path, model)
        np.save(xyz_path, xyz)
        np.save(k_path, intrinsics)


def main():
    parser = argparse.ArgumentParser(description="Run UniDepth over a directory of video frames.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory containing frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write samples-pseudodepth/<frame>.xyz.npy into.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)

    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    model = model.to(torch.device(DEVICE)).eval()

    predict_frames(model, frames_dir, output_dir)


if __name__ == "__main__":
    main()
