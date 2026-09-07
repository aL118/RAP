#!/usr/bin/env python3
"""
Rasterizes one navsim log through navsim's own ScenarioRenderer, so its output
can be compared frame-for-frame against vis3d's custom rasterization of raw
video (vis3d/visualization/raster_frames.py).

The custom pipeline has to *derive* everything it draws -- boxes from
GroundingDINO+SAM masks lifted by UniDepth, traffic-light states read off the
frame pixels, lane markings from YOLOPv2 -- because raw YouTube footage carries
no pose, no calibration and no map. A navsim log carries all of it already, so
this script does no inference at all: it reads the log pkl for ego pose, GT
boxes and traffic-light states, queries the nuPlan HD map for lanes/crosswalks/
walkways, and hands the result to the same ScenarioRenderer the training-time
navsim rasterization uses. Whatever differs between the two outputs is a
difference in the rasterization, not in the input pipeline.

It deliberately does not go through SceneLoader/Scene: those add scene splitting,
sensor loading and filtering that a single log's worth of frames does not need,
and every field Scene._build_ego_status/_build_annotations reads is already a
plain key of the log pkl's per-frame dict. The scenario dict itself is built by
vis3d/visualization/raster.py, shared with the Scene-based path.

Run this in the drivoR env, NOT the vis3d env the rest of scripts/vis3d uses:
querying the HD map pulls in nuplan-devkit and shapely, which vis3d deliberately
does not carry (it is cv2/numpy/tqdm only, so the video pipeline stays torch-free).

Everything else is configured in the CONFIG block below -- no arguments, no env
vars (the maps root is set here rather than read from $NUPLAN_MAPS_ROOT), so it
runs as a bare `python scripts/vis3d/raster_navsim_log.py`.
"""
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

############################### CONFIG ###############################
LOG_PKL = ("/fs/nexus-projects/sim2real/aliu/navsim/dataset/navsim_logs/trainval/"
           "2021.10.05.06.31.40_veh-52_01598_02013.pkl")
SENSOR_BLOBS = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/sensor_blobs/trainval"
MAPS_ROOT = "/fs/nexus-projects/sim2real/aliu/navsim/dataset/maps"
CAMERA = "CAM_F0"

# None = my_dump/navsim_raster/<log>/<camera>. Written under it: vis3d/ (raster on
# black) and vis3d_overlay/ (raster blended onto the photo) -- the same two
# directory names raster_frames.py writes, so the two pipelines line up on disk.
OUTPUT_DIR = "/fs/nexus-projects/sim2real/aliu/RAP/data/navsim"
OVERLAY_ALPHA = 0.6              # raster opacity in vis3d_overlay/

# ScenarioRenderer.observe() hardcodes a 2 m back / 0.8 m up shift of the camera
# (a chase-cam offset that pulls the ego hood into view). True pre-compensates
# for it, so the raster lands on the real camera ray and the overlay lines up
# with the photo; False reproduces navsim's own output, misalignment included.
CANCEL_CAMERA_SHIFT = False

# renderer.py's camera_params is one hardcoded rig for every log, on a 1920x1120
# canvas. False uses this log's real per-frame calibration and its images' actual
# resolution instead, so raster and photo are the same size and the overlay is
# meaningful; True restores the hardcoded pair.
NAVSIM_RIG = False

# Sensor blobs are subsampled relative to the log, and NOT evenly: images come in
# runs of consecutive frames separated by holes of anywhere from 1 to 250 frames.
# Rendering a whole log therefore yields a video that jump-cuts between unrelated
# stretches of road -- which is what makes it look like a short clip on repeat.
# FRAME_RANGE = (first, last) restricts rendering to an inclusive log-frame index
# window, so pick one that lands inside a single long run. None = whole log.
#
# The window below is the longest near-continuous stretch of CAM_F0 coverage in
# trainval: 110 images over log frames 646..760 of this log, no hole wider than 3
# frames, 58 s of real driving covering 449 m. Runner-up in the whole split is 73
# frames, and the longest gapless run anywhere is only 63 -- so 90+ contiguous
# frames does not exist in this data; this window gets there by tolerating holes
# of <=3 frames, invisible at the ~15x speed-up of 2 Hz capture played at 30 fps.
FRAME_RANGE = (646, 760)

# False renders only frames whose image is on disk; True renders every log frame
# in range, with no overlay for the ones that have no image.
ALL_FRAMES = False

LIMIT = None                     # render at most this many frames (None = all)
######################################################################

sys.path.insert(0, str(REPO_ROOT))
# raster.py / raster_frames.py live next to renderer.py and are imported by
# bare name from there (they are scripts, not a package -- no __init__.py).
sys.path.append(str(REPO_ROOT / "vis3d" / "visualization"))

# get_proximal_map_objects casts a column holding NaNs on every map query, once
# per layer per frame -- thousands of identical warnings that bury the progress bar.
warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                        category=RuntimeWarning)

try:
    from pyquaternion import Quaternion  # noqa: E402
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

    from raster import build_scenario_dict_from_values  # noqa: E402
    from raster_frames import overlay_on_frame  # noqa: E402
    from navsim.visualization.renderer import ScenarioRenderer, camera_params  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"Cannot import {exc.name!r}. This script needs the full navsim stack (see the "
        "module docstring): run it in the drivoR env, `conda activate drivoR`, not vis3d."
    ) from exc

# observe()'s hardcoded chase-cam offset, in the lidar frame: cam_t[0] -= 2, cam_t[2] += 0.8.
OBSERVE_CAMERA_SHIFT = np.array([-2.0, 0.0, 0.8])


def ego_pose_of(frame: dict) -> np.ndarray:
    """Global (x, y, heading) of the ego, the same way Scene._build_ego_status derives it."""
    translation = frame["ego2global_translation"]
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def camera_model_of(frame: dict) -> dict:
    """
    ScenarioRenderer camera model for CAMERA: the log's own calibration unless
    NAVSIM_RIG (the hardcoded rig belongs to a different vehicle, so boxes drawn
    through it sit a few pixels off this log's images), keyed as observe()
    expects -- note the log calls the intrinsics 'cam_intrinsic'.
    """
    if NAVSIM_RIG:
        model = {k: np.asarray(v).copy() for k, v in camera_params[CAMERA].items()}
    else:
        cam = frame["cams"][CAMERA]
        model = {
            "sensor2lidar_translation": np.asarray(cam["sensor2lidar_translation"], np.float64).copy(),
            "sensor2lidar_rotation": np.asarray(cam["sensor2lidar_rotation"], np.float64),
            "intrinsics": np.asarray(cam["cam_intrinsic"], np.float64),
        }
    if CANCEL_CAMERA_SHIFT:
        model["sensor2lidar_translation"] -= OBSERVE_CAMERA_SHIFT
    return model


def main():
    log_pkl = Path(LOG_PKL)
    sensor_blobs = Path(SENSOR_BLOBS)
    with open(log_pkl, "rb") as f:
        log_frames = pickle.load(f)

    output_dir = (Path(OUTPUT_DIR) if OUTPUT_DIR
                  else REPO_ROOT / "my_dump" / "navsim_raster" / log_pkl.stem / CAMERA)
    vis_dir = output_dir / "vis3d"
    overlay_dir = output_dir / "vis3d_overlay"
    vis_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # One map_api per location: get_maps_api caches, but a log only ever visits one.
    map_apis = {}

    if FRAME_RANGE is not None:
        first, last = FRAME_RANGE
        log_frames = log_frames[first:last + 1]

    selected = []
    for frame in log_frames:
        image_path = sensor_blobs / frame["cams"][CAMERA]["data_path"]
        if image_path.exists():
            selected.append((frame, image_path))
        elif ALL_FRAMES:
            selected.append((frame, None))
    if LIMIT is not None:
        selected = selected[:LIMIT]
    if not selected:
        raise SystemExit(f"No {CAMERA} frames found under {sensor_blobs} for {log_pkl.name}")

    n_overlays = 0
    for seq, (frame, image_path) in enumerate(tqdm(selected, desc=f"rasterizing {CAMERA}")):
        location = frame["map_location"]
        if location not in map_apis:
            map_apis[location] = get_maps_api(MAPS_ROOT, "nuplan-maps-v1.0", location)

        scenario = build_scenario_dict_from_values(
            map_apis[location],
            ego_pose_of(frame),
            frame["anns"]["gt_boxes"],
            frame["anns"]["gt_names"],
            frame["traffic_lights"],
        )

        frame_bgr = cv2.imread(str(image_path)) if image_path is not None else None
        if NAVSIM_RIG or frame_bgr is None:
            height, width = 1120, 1920
        else:
            height, width = frame_bgr.shape[:2]

        renderer = ScenarioRenderer(camera_channel_list=[CAMERA], width=width, height=height)
        renderer.camera_models = {CAMERA: camera_model_of(frame)}
        # observe() returns RGB (its palettes are RGB triples handed to cv2.fillConvexPoly),
        # so swap before writing -- same convention as raster_frames.py.
        canvas_bgr = renderer.observe(scenario)[CAMERA][:, :, ::-1]

        # Sensor images are named by content hash, which sorts arbitrarily -- stitching
        # them with frames_to_video.py (sorted(glob)) would shuffle the drive into a
        # jumble that reads as one short clip on repeat. Prefixing the temporal index
        # makes lexicographic order the driving order; the hash is kept so a rendered
        # frame can still be traced back to its source jpg.
        name = f"{seq:06d}_{Path(frame['cams'][CAMERA]['data_path']).name}"
        cv2.imwrite(str(vis_dir / name), canvas_bgr)
        if frame_bgr is not None:
            cv2.imwrite(str(overlay_dir / name),
                        overlay_on_frame(canvas_bgr, frame_bgr, OVERLAY_ALPHA))
            n_overlays += 1

    print(f"Saved {len(selected)} rendered frames to {vis_dir}")
    print(f"Saved {n_overlays} overlay frames to {overlay_dir}")


if __name__ == "__main__":
    main()
