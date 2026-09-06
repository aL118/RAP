"""
Estimates the ego trajectory of a clip with OpenVO (vendored under openvo/),
writing KITTI-format poses that export_navsim_logs.py --poses turns into
navsim's ego2global_* fields.

This is the stage that makes video clips usable for trajectory supervision:
without it a clip has no ego motion at all, and navsim's training target --
Scene.get_future_trajectory, which reads nothing but ego poses -- would be a
row of zeros.

Depth comes from this pipeline's existing UniDepth point maps rather than from
OpenVO's own depth stage. Upstream, OpenVO runs Metric3D for depth and
WildCamera for intrinsics (processor/depth_processor.py, a third conda env and
2.7 GB of weights); here infer_unidepth_on_frames.py has already produced a
metric point map and intrinsics for every frame, and reusing them means the
trajectory, the boxes and the lanes are all derived from one camera and one
depth model instead of two that can disagree.

Runs in the 'vis3d' env like every other stage. The model's correlation layer
is a compiled CUDA extension (openvo/model/correlation_package); it used to be
distributed as a cpython-39 .so, which is what kept this stage in its own
py3.9/torch-2.0.1 env. It is now rebuilt against this env's torch -- the source
needed only a c++14 -> c++17 bump, see
scripts/vis3d/openvo-correlation-cxx17.patch.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
DEFAULT_WEIGHTS = ROOT.parent / "weights" / "openvo" / "model_ep-022.pt"

# OpenVO's depth PNGs are metres * 256 as uint16 -- the convention its own
# depth_processor.save_depth_png writes and its dataset reads back. The reader
# then rescales again (dataset.py: / 200.0 / 300.0), so what matters is only
# that this matches what the model was trained on, not that the stored numbers
# are metres.
DEPTH_SCALE = 256.0
DEPTH_MAX_UINT16 = 65535


def write_openvo_depth(xyz_dir: Path, frame_names: list, depth_dir: Path, scene: str,
                       intrinsics: np.ndarray) -> None:
    """Converts UniDepth point maps into the depth PNGs + intrinsics json that
    OpenVO's dataset expects.

    The intrinsics file goes next to the scene's depth folder as
    <scene>_intrs.json, which is where InferenceVO looks for it, and holds
    [fx, fy, cx, cy] at the frames' own resolution.
    """
    depth_dir.mkdir(parents=True, exist_ok=True)
    for name in tqdm(frame_names, desc="depth"):
        xyz = np.load(xyz_dir / f"{Path(name).stem}.xyz.npy")
        # z of the camera-frame point map is depth along the optical axis,
        # which is what a depth map is.
        depth = np.clip(xyz[2] * DEPTH_SCALE, 0, DEPTH_MAX_UINT16).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{Path(name).stem}.png"), depth)

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    with open(depth_dir.parent / f"{scene}_intrs.json", "w") as file:
        json.dump({scene: [fx, fy, cx, cy]}, file, indent=2)


def run_openvo(frames_dir: Path, depth_dir: Path, scene: str, weights: Path,
               save_path: Path) -> Path:
    """Runs the vendored OpenVO inference, returning its pose file.

    Imported rather than shelled out to, but its modules import each other
    flatly ('from model import *'), so openvo/ has to be on sys.path -- the
    same arrangement visualization/raster_frames.py has with renderer.py.
    """
    openvo_dir = ROOT / "openvo"
    sys.path.insert(0, str(openvo_dir))
    from my_inference import InferenceVO  # noqa: E402

    config = openvo_dir / "configs" / "drivor.py"
    # InferenceVO.__init__ calls read_config(), which re-parses sys.argv for a
    # single required --config; this module's own arguments must not reach it.
    sys.argv = [sys.argv[0], "--config", str(config)]

    inference = InferenceVO(
        root_path=str(ROOT / "openvo"),
        save_path=str(save_path),
        weight=str(weights),
        nprocess=1,
        frames_dir=str(frames_dir),
        depth_dir=str(depth_dir),
        save_name=scene,
    )
    if not inference.already:
        processes, _ = inference.launch_workers(
            run_py=str(openvo_dir / "my_inference_utils.py"),
            devices=[0],
            python_bin=sys.executable,
        )
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("OpenVO inference worker failed; see its output above.")

    # results/<key>/<weight dir>/<save name>/<scene>.txt
    pose_files = sorted((save_path / "YouTube" / weights.parent.name / scene).glob("*.txt"))
    if not pose_files:
        raise FileNotFoundError(
            f"OpenVO wrote no trajectory under {save_path}; the worker may have failed silently.")
    return pose_files[0]


def _static_mask(masks_json, frame: str, shape) -> np.ndarray:
    """Boolean mask, True where the pixel is *not* a detected vehicle,
    pedestrian or other movable object.

    Point-cloud odometry measures the motion of whatever it tracks, so on a
    queue of traffic -- which fills most of the frame in exactly the clips we
    care about -- tracking everything measures the traffic instead of the ego.
    Detections are dilated a little because a segmentation mask stops at the
    object's edge, while the depth around that edge is already contaminated.
    """
    mask = np.ones(shape, bool)
    entries = masks_json.get(frame, [])
    if not entries:
        return mask
    from pycocotools import mask as mask_utils  # optional: only this path needs it

    for entry in entries:
        decoded = mask_utils.decode(entry).astype(bool)
        if decoded.shape != shape:
            decoded = cv2.resize(decoded.astype(np.uint8), (shape[1], shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        mask &= ~decoded
    kernel = np.ones((9, 9), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _load_object_masks(output_dir: Path) -> dict:
    """mask_results_preds.json grouped by frame, or {} if it isn't there."""
    path = output_dir / "mask_results_preds.json"
    if not path.exists():
        print(f"No {path.name}; tracking the whole frame, including any moving vehicles.")
        return {}
    with open(path) as file:
        detections = json.load(file)
    grouped = {}
    for detection in detections:
        grouped.setdefault(detection["frame"], []).append(detection["mask"])
    return grouped


def run_point_cloud(frames_dir: Path, frame_names: list, xyz_dir: Path,
                    output_dir: Path) -> np.ndarray:
    """Geometric odometry from the point maps -- no learned model, no GPU."""
    from point_cloud_odometry import estimate_trajectory  # noqa: E402

    masks_json = _load_object_masks(output_dir)
    shape = np.load(xyz_dir / f"{Path(frame_names[0]).stem}.xyz.npy").shape[1:]
    static_masks = [_static_mask(masks_json, name, shape) for name in tqdm(frame_names, desc="masks")]

    return estimate_trajectory(
        [frames_dir / name for name in frame_names],
        [xyz_dir / f"{Path(name).stem}.xyz.npy" for name in frame_names],
        static_masks=static_masks,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a clip's ego trajectory, for export_navsim_logs.py --poses.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory of frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Run directory holding samples-pseudodepth/; ego_poses.txt is written here.")
    parser.add_argument("--method", choices=("pointcloud", "openvo"), default="pointcloud",
                        help="pointcloud: register consecutive UniDepth point maps (CPU, metric by "
                             "construction). openvo: the learned model in openvo/ (GPU); its released "
                             "checkpoints were trained on LiDAR depth and under-predict motion on "
                             "dashcam footage by roughly 20x.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="OpenVO checkpoint.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene label for OpenVO's outputs (default: the frames' clip name).")
    parser.add_argument("--keep_depth", action="store_true",
                        help="Keep the intermediate OpenVO-format depth PNGs instead of deleting them.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    scene = args.scene or frames_dir.parent.name

    xyz_dir = output_dir / "samples-pseudodepth"
    if not xyz_dir.is_dir():
        raise FileNotFoundError(
            f"{xyz_dir} does not exist; run infer_unidepth_on_frames.py first -- "
            "the trajectory is estimated from its point maps.")

    frame_names = sorted(p.name for p in frames_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                         and (xyz_dir / f"{p.stem}.xyz.npy").exists())
    assert frame_names, f"No frames in {frames_dir} have a point map in {xyz_dir}"

    sys.path.insert(0, str(ROOT))
    from lift_frames_to_3d import _camera_intrinsics  # noqa: E402

    first = np.load(xyz_dir / f"{Path(frame_names[0]).stem}.xyz.npy")
    # One camera for the whole clip: the intrinsics are the same physical lens
    # every frame, and OpenVO's config carries a single [fx, fy, cx, cy] per
    # scene rather than a per-frame value.
    intrinsics = _camera_intrinsics(first, xyz_dir / f"{Path(frame_names[0]).stem}.K.npy")

    destination = output_dir / "ego_poses.txt"
    if args.method == "openvo":
        work_dir = output_dir / "openvo"
        depth_dir = work_dir / "depth" / scene
        write_openvo_depth(xyz_dir, frame_names, depth_dir, scene, intrinsics)
        pose_file = run_openvo(frames_dir, depth_dir, scene, Path(args.weights).resolve(),
                               work_dir / "results")
        shutil.copy2(pose_file, destination)
        if not args.keep_depth:
            shutil.rmtree(work_dir / "depth", ignore_errors=True)
    else:
        trajectory = run_point_cloud(frames_dir, frame_names, xyz_dir, output_dir)
        # Same KITTI layout OpenVO writes, so both methods feed
        # export_navsim_logs.py --poses unchanged.
        np.savetxt(destination, trajectory[:, :3, :].reshape(len(trajectory), 12))

    poses = np.loadtxt(destination).reshape(-1, 12)
    if len(poses) != len(frame_names):
        print(f"Warning: {len(poses)} poses for {len(frame_names)} frames; "
              "export_navsim_logs.py indexes poses by frame number, so a mismatch "
              "means some frames will have no pose.")

    print(f"Wrote {len(poses)} ego poses to {destination}")


if __name__ == "__main__":
    main()
