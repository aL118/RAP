"""
Writes a vis3d run (lift_frames_to_3d.py's boxes_3d.json + the frames it was
built from) as a navsim-format log, so video clips can be loaded through the
same SceneLoader/Scene API as the nuPlan trainval data.

The template is not the nuPlan logs but this repo's CARLA sim logs
(carla_data_split/openscene_meta_datas/*.pkl): they are already navsim-format
logs for data with no nuPlan map, which is exactly our situation.
navsim.common.dataclasses._build_map_api returns None for a map_location it
does not recognise, so a clip's own name can go in that field.

What this deliberately does NOT provide is ego motion. A dashcam clip carries
no odometry, so ego2global_* and ego_dynamic_state are written as zeros and
flagged with has_ego_pose=False. Consumers that only rasterize (boxes, lights
and lanes are all in the ego frame) are unaffected, but Scene.get_future_
trajectory builds its label purely from those pose fields, so these logs must
not be used for trajectory supervision -- it would train on a label that says
the car never moves. Adding visual odometry is what would change that, and
only these fields would need to change.

Traffic lights and lanes are written under extra keys rather than being forced
into the schema. navsim's own "traffic_lights" field is a list of
(lane_connector_id, is_red) -- a *map reference* resolved to a position at
render time -- which cannot express a detected light's measured position or a
three-way state, and there is no lane field at all (lane geometry comes from
map_api). The loader reads a fixed set of keys, so the extra ones ride along
harmlessly and visualization/raster_frames.py's data is preserved exactly.

Usage:
    python export_navsim_logs.py --output_dir <run dir> --frames_dir <frames> \
        --dataset_root <where to write> [--log_name NAME] [--split video]
"""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

import box_schema
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]

CAMERA_CHANNEL = "CAM_F0"

# navsim frames are 2 Hz -- navsim.common.dataclasses.NAVSIM_INTERVAL_LENGTH is
# 0.5 s and is baked into every Trajectory returned by get_future_trajectory,
# and the scene filters count on it ("frames are at 2Hz", all_scenes.yaml).
# Clips come out of process_ytb.py at 10 Hz by default, so frames are dropped
# on export to match. Exporting at the source rate does not fail, it silently
# mislabels time: a 10-frame future would be 1 s of footage that every
# consumer treats as 5 s, and the trajectory targets would be off by 5x.
NAVSIM_HZ = 2.0
FRAME_INTERVAL_US = int(1e6 / NAVSIM_HZ)

# The rate process_ytb.py extracted the frames at (its --hz).
DEFAULT_SOURCE_HZ = 10.0

# One-hot [left, straight, right, unknown]. A clip has no route, so the command
# is unknown -- the same value the CARLA logs use when there is no route.
DRIVING_COMMAND = np.array([0, 0, 0, 1], dtype=int)

# nuPlan's lidar sits behind and above the rear axle; for a dashcam clip the
# "lidar" frame is just the ego frame lift_frames_to_3d.py works in, so this is
# identity. Kept because the schema has the fields and consumers index them.
LIDAR2EGO_ROTATION = np.array([1.0, 0.0, 0.0, 0.0])
LIDAR2EGO_TRANSLATION = np.zeros(3)

# Camera (x right, y down, z forward) -> ego (x forward, y left, z up), for a
# camera mounted perfectly level. The fallback only: the real rotation is read
# off boxes_3d.json's ego_to_camera, which carries whatever attitude
# calibrate_ground measured, so that the trajectory and the boxes land in the
# same frame. Using this constant on a clip whose camera looks 7 deg down (as
# wrongway/4's does) tilts the whole trajectory into the road.
CAMERA_TO_EGO_AXES = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])


def _camera_axes(boxes_by_frame: dict) -> np.ndarray:
    """The clip's camera -> ego rotation, from the boxes' own transform."""
    for entry in boxes_by_frame.values():
        camera = entry.get("camera")
        if camera:
            return np.linalg.inv(
                np.asarray(camera["ego_to_camera"], dtype=np.float64))[:3, :3]
    return CAMERA_TO_EGO_AXES


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a 3x3 rotation, the order navsim's Quaternion(*...) expects.

    Shepperd's method: pick the branch whose denominator is largest, so no
    near-zero division on any rotation.
    """
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def load_ego_poses(poses_path: Path, cam_to_ego: np.ndarray = None) -> np.ndarray:
    """Reads OpenVO's trajectory file as (N, 4, 4) ego-to-global transforms.

    OpenVO (slurm/inference_full.sh -> src/my_inference.py) writes KITTI-format
    poses: one line per frame, 12 numbers, a row-major 3x4 camera-to-world
    matrix, starting from identity. Its frames come from the same
    split_video_by_hz_precise() as this pipeline's, at the same default 10 Hz,
    so line i corresponds to frame i -- which is why frames are matched to
    poses by their numeric filename below rather than by position in a list.

    The poses are in camera axes; the ego convention is that remap composed with
    the mount's attitude, so both the translation and (by conjugation) the
    rotation are converted here. `cam_to_ego` defaults to the level camera.
    """
    axes = CAMERA_TO_EGO_AXES if cam_to_ego is None else np.asarray(cam_to_ego)
    rows = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = axes @ rows[:, :3, :3] @ axes.T
    poses[:, :3, 3] = rows[:, :3, 3] @ axes.T
    return poses


def _dynamic_state(poses: np.ndarray, index: int, interval: float) -> list:
    """[vx, vy, ax, ay] in the ego frame, by finite differences of `poses`.

    Central differences where possible, one-sided at the ends. Velocity is
    rotated into the current frame's own axes, which is what EgoStatus means by
    ego_velocity (navsim.common.dataclasses._build_ego_status).
    """
    def velocity(i: int) -> np.ndarray:
        previous, following = max(i - 1, 0), min(i + 1, len(poses) - 1)
        if following == previous:
            return np.zeros(3)
        world = (poses[following, :3, 3] - poses[previous, :3, 3]) / ((following - previous) * interval)
        return poses[i, :3, :3].T @ world  # world -> this frame's axes

    current = velocity(index)
    if index + 1 < len(poses):
        acceleration = (velocity(index + 1) - current) / interval
    elif index > 0:
        acceleration = (current - velocity(index - 1)) / interval
    else:
        acceleration = np.zeros(3)
    return [float(current[0]), float(current[1]),
            float(acceleration[0]), float(acceleration[1])]


def _token(*parts: str) -> str:
    """A stable 16-hex-character id, the shape navsim tokens have.

    Derived from the inputs rather than randomly, so re-exporting a clip
    produces the same tokens and anything keyed on them stays valid.
    """
    return hashlib.md5("/".join(parts).encode()).hexdigest()[:16]


def _camera_entry(camera: dict, data_path: str) -> dict:
    """The cams[CAM_F0] dict, from boxes_3d.json's per-frame camera.

    sensor2lidar_* is the inverse of the ego->camera transform the boxes were
    fit through: navsim stores the camera's pose in the ego/lidar frame, which
    is the opposite direction from what lift_frames_to_3d.py records.
    """
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    camera_to_ego = np.linalg.inv(ego_to_camera)
    return {
        "data_path": data_path,
        "cam_intrinsic": np.asarray(camera["intrinsics"], dtype=np.float64),
        # UniDepth predicts a pinhole camera, so there is no distortion model
        # to carry: the point map and the intrinsics recovered from it are
        # already consistent with an undistorted pinhole.
        "distortion": np.zeros(5),
        "sensor2lidar_rotation": camera_to_ego[:3, :3],
        "sensor2lidar_translation": camera_to_ego[:3, 3],
    }


def _annotations(entry: dict) -> dict:
    """The anns dict. Boxes come out of box_schema as
    [x, y, z, l, w, h, roll, pitch, yaw] in the ego frame, which is navsim's own
    convention (see visualization/raster.py's _boxes_ego_to_world) widened by
    the two extra angles."""
    boxes = box_schema.to_array(entry["boxes"])
    count = len(boxes)
    return {
        "gt_boxes": boxes,
        "gt_names": np.asarray(entry["names"], dtype="<U32"),
        # No tracking across frames, so no velocity can be measured. Zeros are
        # the honest value; nothing in the rasterization path reads them.
        "gt_velocity_3d": np.zeros((count, 3), dtype=np.float64),
        # Detections are per-frame, so an "instance" exists for one frame only
        # and its token cannot be shared with the same object in the next one.
        "instance_tokens": [_token("instance", str(i)) for i in range(count)],
        "track_tokens": [_token("track", str(i)) for i in range(count)],
    }


def build_frames(boxes_by_frame: dict, frame_names: list, log_name: str,
                 blob_relative_dir: str, ego_poses: np.ndarray = None,
                 source_interval: float = None) -> list:
    """Builds the navsim frame-dict list for one clip, in temporal order.

    :param ego_poses: optional (N, 4, 4) ego-to-global transforms indexed by the
        frame's own number (000123.jpg -> ego_poses[123]), from load_ego_poses.
        Without them the pose fields are zeroed and has_ego_pose is False.
    :param source_interval: seconds between consecutive *source* frames, which
        is what ego_poses is sampled at -- velocity is differenced there, not
        across the exported 2 Hz frames, so dropping frames does not change it.
    """
    log_token = _token("log", log_name)
    scene_token = _token("scene", log_name)
    tokens = [_token(log_name, name) for name in frame_names]

    frames = []
    for index, name in enumerate(frame_names):
        entry = boxes_by_frame[name]

        pose = np.eye(4)
        dynamic_state = [0.0, 0.0, 0.0, 0.0]
        if ego_poses is not None:
            source_index = int(Path(name).stem)
            pose = ego_poses[source_index]
            dynamic_state = _dynamic_state(ego_poses, source_index, source_interval)
        frames.append({
            "token": tokens[index],
            "frame_idx": index,
            "timestamp": index * FRAME_INTERVAL_US,
            "log_name": log_name,
            "log_token": log_token,
            "scene_name": log_name,
            "scene_token": scene_token,
            "vehicle_name": "dashcam",
            # Unknown to _build_map_api, which is what makes it return None
            # instead of trying to load a nuPlan map for this clip.
            "map_location": log_name,
            "sample_prev": tokens[index - 1] if index > 0 else None,
            "sample_next": tokens[index + 1] if index + 1 < len(frame_names) else None,

            "anns": _annotations(entry),
            "cams": {CAMERA_CHANNEL: _camera_entry(
                entry["camera"], f"{blob_relative_dir}/{CAMERA_CHANNEL}/{name}")},

            # --- ego motion: from OpenVO if given, else zeroed (see docstring) ---
            "has_ego_pose": ego_poses is not None,
            "ego2global_translation": pose[:3, 3],
            "ego2global_rotation": _quaternion_from_matrix(pose[:3, :3]),
            "ego2global": pose,
            "ego_dynamic_state": dynamic_state,  # vx, vy, ax, ay
            "can_bus": np.zeros(18),
            "driving_command": DRIVING_COMMAND,

            "lidar2ego_rotation": LIDAR2EGO_ROTATION,
            "lidar2ego_translation": LIDAR2EGO_TRANSLATION,
            "lidar2ego": np.eye(4),
            "lidar2global": np.eye(4),
            "lidar_path": None,

            # Map-derived fields with no source for a video clip. Empty
            # roadblock_ids means these scenes only load under a scene filter
            # with has_route=False (as the CARLA logs also require).
            "roadblock_ids": [],
            "traffic_lights": [],
            "flow_gt_final_path": None,
            "occ_gt_final_path": None,

            # --- vis3d extras, ignored by stock navsim ---
            "traffic_lights_3d": entry.get("traffic_lights", []),
            "lanes": entry.get("lanes", []),
        })
    return frames


def _subsample(frame_names: list, source_hz: float, phase: int) -> list:
    """Drops frames down to NAVSIM_HZ, keeping every `stride`-th one.

    `phase` selects which of the `stride` interleavings to keep. Each phase is
    an equally valid 2 Hz sequence of the same clip, so exporting several of
    them (under different log names) multiplies the scene count -- at the cost
    of scenes that overlap in content.
    """
    stride = int(round(source_hz / NAVSIM_HZ))
    if stride < 1:
        raise ValueError(f"--source_hz {source_hz} is below navsim's {NAVSIM_HZ} Hz; "
                         "frames cannot be invented, re-extract the clip at a higher rate.")
    if not 0 <= phase < stride:
        raise ValueError(f"--phase must be in [0, {stride}) for source_hz={source_hz}")
    return frame_names[phase::stride]


def export(output_dir: Path, frames_dir: Path, dataset_root: Path, log_name: str,
           split: str, link_sensors: bool, source_hz: float, phase: int,
           poses_path: Path = None) -> Path:
    with open(output_dir / "boxes_3d.json") as file:
        boxes_by_frame = json.load(file)

    frame_names = sorted(name for name in boxes_by_frame if (frames_dir / name).exists())
    if not frame_names:
        raise FileNotFoundError(
            f"None of boxes_3d.json's frames were found in {frames_dir}; "
            "--frames_dir must be the directory the run was built from.")
    missing = len(boxes_by_frame) - len(frame_names)
    if missing:
        print(f"Warning: {missing} frames in boxes_3d.json are not in {frames_dir}; skipping them.")

    available = len(frame_names)
    frame_names = _subsample(frame_names, source_hz, phase)
    print(f"Kept {len(frame_names)} of {available} frames "
          f"({source_hz:g} Hz -> {NAVSIM_HZ:g} Hz, phase {phase}).")

    log_dir = dataset_root / "navsim_logs" / split
    blob_dir = dataset_root / "sensor_blobs" / split / log_name / CAMERA_CHANNEL
    log_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    for name in tqdm(frame_names, desc="sensors"):
        destination = blob_dir / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link_sensors:
            destination.symlink_to(frames_dir / name)
        else:
            shutil.copy2(frames_dir / name, destination)

    ego_poses = None
    if poses_path is not None:
        ego_poses = load_ego_poses(poses_path, _camera_axes(boxes_by_frame))
        needed = max(int(Path(name).stem) for name in frame_names) + 1
        if len(ego_poses) < needed:
            raise ValueError(
                f"{poses_path} has {len(ego_poses)} poses but frame "
                f"{needed - 1:06d} needs one. The pose file must come from the same "
                "video extracted at the same rate as --frames_dir.")
        print(f"Ego poses: {len(ego_poses)} from {poses_path}")

    frames = build_frames(boxes_by_frame, frame_names, log_name, log_name,
                          ego_poses=ego_poses, source_interval=1.0 / source_hz)
    log_path = log_dir / f"{log_name}.pkl"
    with open(log_path, "wb") as file:
        pickle.dump(frames, file)

    lights = sum(len(frame["traffic_lights_3d"]) for frame in frames)
    lanes = sum(len(frame["lanes"]) for frame in frames)
    boxes = sum(len(frame["anns"]["gt_boxes"]) for frame in frames)
    print(f"Wrote {len(frames)} frames ({boxes} boxes, {lights} traffic lights, "
          f"{lanes} lane polylines) to {log_path}")
    print(f"Sensor images: {blob_dir}")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a vis3d run as a navsim-format log.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Run directory containing boxes_3d.json.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory of the original frame images.")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root to write navsim_logs/<split>/ and sensor_blobs/<split>/ into.")
    parser.add_argument("--log_name", type=str, default=None,
                        help="Name of the log (default: the clip's directory name).")
    parser.add_argument("--split", type=str, default="video",
                        help="Split directory name under navsim_logs/ and sensor_blobs/.")
    parser.add_argument("--copy_sensors", action="store_true",
                        help="Copy frame images instead of symlinking them.")
    parser.add_argument("--source_hz", type=float, default=DEFAULT_SOURCE_HZ,
                        help="Rate the frames were extracted at (process_ytb.py's --hz). "
                             f"Frames are dropped to navsim's {NAVSIM_HZ:g} Hz on export.")
    parser.add_argument("--poses", type=str, default=None,
                        help="OpenVO trajectory file for this clip "
                             "(OpenVO/results/YouTube/openvo_nusc_gt/<scene>/<scene>.txt). "
                             "Without it the log carries no ego motion and must not be used "
                             "for trajectory supervision.")
    parser.add_argument("--phase", type=int, default=0,
                        help="Which interleaving to keep when subsampling (0 <= phase < stride). "
                             "Export several phases under different --log_name values to get "
                             "more scenes out of one clip.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    frames_dir = Path(args.frames_dir).resolve()
    # Default: the clip directory, i.e. the parent of a run subdir like "2",
    # or the run dir itself when the run wrote straight into the clip dir.
    log_name = args.log_name or (
        output_dir.parent.name if output_dir.name.isdigit() else output_dir.name)

    export(output_dir, frames_dir, Path(args.dataset_root).resolve(), log_name,
           args.split, link_sensors=not args.copy_sensors,
           source_hz=args.source_hz, phase=args.phase,
           poses_path=Path(args.poses).resolve() if args.poses else None)


if __name__ == "__main__":
    main()
