"""
Write the vis3d rasterization runs out as navsim-format logs.

Input is what the vis3d pipeline already produced for a clip:

  data/<dataset>/<clip>/
    frames/                      the extracted video frames
    <run>/boxes_3d.json          lifted boxes + lanes + lights + per-frame camera
    <run>/vis3d/                 the rasterized render of each frame
    <run>/ego_poses.txt          OpenVO / point-cloud odometry (estimate_ego_motion.py)

Output mirrors carla_garage_data_navsim_converted -- the layout
run_training_full.py's sim_log_path / sim_sensor_path already point at -- so
these clips load through the same SceneLoader/Scene API as the CARLA logs:

  <DATASET_ROOT>/
    openscene_meta_datas/<scene_token>.pkl        list[dict], one dict per frame
    sensor_blobs/<log_name>/CAM_F0/<token>.jpg    the photograph
    rendered_sensor_blobs/<log_name>/CAM_F0/<token>.jpg
                                                  the rasterization of it
    synthetic_scene_pickles/<scene_token>.pkl     two-stage eval stub, see below

RAP reads BOTH image trees, as a pair. It is never told where the second one
is: navsim/common/dataclasses.py:77 derives that path from the first by string
substitution,

    str(image_path).replace('sensor_blobs', 'rendered_sensor_blobs')

so the two trees have to agree filename for filename -- same log, same channel,
same <token>.jpg -- or the rasterization simply does not reach the model. Nor
are they two views fed side by side: agent_lightning_module.py:159 does

    features['camera_feature'] = features.pop('rendered_camera_feature')

so the rasterization *becomes* the camera feature on the training branch, while
the photograph feeds a second branch gated by real_valid. The rendered tree is
the one that always runs, which is why this script writes both rather than
choosing between them.

Every way of getting that wrong is silent. dataclasses.py:79-82 swallows an
unreadable rendered image into np.zeros((1080,1920,3)) behind a bare except,
and a missing real image becomes zeros_like with real_valid False. A tree that
is misnamed, short a frame, or absent altogether trains on black images and
reports nothing at all.

Everything goes straight into that one tree rather than into the run directory,
because navsim's SceneLoader takes a single data_path and unpickles every file
in it: one directory per clip would need one loader per clip.

Structured after scripts/data/convert_carla_to_navsim.py: one get_<field>()
per key of the frame dict, called in the dict literal, so the schema reads top
to bottom in one place. Those functions take no arguments -- the per-frame
inputs live in the module-level FRAME context that build_frame() fills in, and
everything else is a constant in the CONFIG block below. There is no CLI:
change the constants and re-run. The one exception is --dry-run, which prints
the frames it would write instead of writing, linking or copying anything.

Run this in the *training* env (drivoR), not the vis3d one. It needs nothing
but numpy and the stdlib, and the numpys differ across the two: vis3d is on
2.4, drivoR on 1.24, and a pickle written by the former dies in the latter with
"No module named 'numpy._core'" -- at log-load time, inside the training job,
long after this script reported success.

What is deliberately NOT here:
  - lidar. lidar_path is None and the lidar2* fields are identity. The drivoR
    agent's config sets lidar_pc: [], so nothing loads them.
  - maps. map_location is the clip's own name, which navsim's _build_map_api
    does not recognise, so it returns None instead of trying to load a nuPlan
    map. roadblock_ids is therefore empty, and these logs only survive a scene
    filter with has_route: false -- the same requirement the CARLA logs have.
  - navsim's traffic_lights, which is a list of (lane_connector_id, is_red):
    a map reference, not something a detector can produce. The measured lights
    ride along under traffic_lights_3d instead, with the lanes, where stock
    navsim ignores them and visualization/raster_frames.py can still read them.
  - a real synthetic_scene_pickles entry. Training never reads that directory
    (only navhard's two-stage evaluation does, out of its own NAVHARD_DATA_ROOT),
    so what is written is the same near-empty stub the CARLA conversion writes:
    scene_metadata plus three empty lists. WRITE_SYNTHETIC_SCENES turns it off.
"""

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

# box_schema defines the boxes_3d.json row layout; vis3d/ is not a package,
# so it goes on the path the same way the other cross-tree imports here do.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis3d"))
import box_schema

# ---------------------------------------------------------------------------
# CONFIG -- hardcoded on purpose; edit here rather than passing flags
# ---------------------------------------------------------------------------

BASE = Path("/fs/nexus-projects/sim2real/aliu/RAP")

# The one tree every clip is written into, laid out exactly like
# carla_garage_data_navsim_converted so a training run picks it up with
#   sim_log_path:    <DATASET_ROOT>/openscene_meta_datas
#   sim_sensor_path: <DATASET_ROOT>/sensor_blobs
DATASET_ROOT = BASE / "CARE"
META_DIR_NAME = "openscene_meta_datas"
BLOB_DIR_NAME = "sensor_blobs"
# Not a free choice: dataclasses.py:77 builds this path by substituting this
# name into the sensor_blobs one. Renaming either breaks the pairing silently.
RENDERED_BLOB_DIR_NAME = "rendered_sensor_blobs"
SYNTHETIC_DIR_NAME = "synthetic_scene_pickles"

# Write the synthetic_scene_pickles stub alongside each log. Off costs nothing
# for training, which never opens that directory; on keeps the tree the same
# shape as the CARLA one, so anything that walks all three directories in step
# does not trip over a missing file.
WRITE_SYNTHETIC_SCENES = True

# (dataset, clip, run subdir) triples. `dataset` is the directory under data/,
# `run` the output subdir inside the clip -- "" means the run wrote straight
# into the clip directory.
#
# A clip of "*" stands for every clip directory in that dataset rasterized at
# that run, which is how a whole batch goes out without naming its clips one at
# a time. Clips in the dataset that have not been rendered at that run are
# listed and skipped rather than exported as empty logs; see expand_clip_runs.
#
# beepbeep/2, NOT beepbeep/1: run 1's point maps were back-projected through a
# supplied fx of 806 px instead of UniDepth's own ~2138, which fans the scene
# out sideways (x spanning -18..+136 m in a single frame) and leaves every box
# sheared. Run 2 is the same clip re-rendered with no supplied camera and is
# the one whose metric fields mean anything.
#
# The other three YTB clips were lifted with that same wrong camera and have to
# be re-rendered before they can be listed here:
#   ("YTB", "snowcrash", "smooth"), ("YTB", "changelane", "smooth"),
#   ("YTB", "redlight", "smooth")
CLIP_RUNS = [
    ("CARE_YTB", "*", "1"),
]

# Which images become each tree. Both are written on every run.
#
#   "frames"        the untouched photograph
#   "vis3d"         the rasterized render
#   "vis3d_overlay" the render blended over the photograph -- for eyeballing
#                   only, and for neither tree: no real camera looks like that,
#                   and it is not a clean rasterization either
REAL_SENSOR_SOURCE = "frames"
RENDERED_SENSOR_SOURCE = "vis3d"

# Rows of black added to the top AND bottom of every rendered frame.
#
# navsim's own rendered views are 40 px taller than the matching photograph
# (1920x1120 against 1920x1080) because dataclasses.py:81 reads them as
# Image.open(...)[20:-20]. That crop is unconditional. A rendered image written
# at the photograph's height therefore loses 20 px off each end and every row
# that survives sits 20 px from where the real image puts it -- a registration
# error between the two streams that nothing downstream measures or reports.
#
# vis3d renders at exactly the frame size, so the margin is added here. Set this
# to 0 if the renderer is ever changed to emit the taller canvas itself, or the
# frame gets padded twice.
RENDERED_PAD_ROWS = 20

# Quality for the re-encoded rendered frames. Padding means a decode and
# re-encode rather than a link, and PIL's default of 75 would visibly degrade an
# image the model then treats as ground truth. Unused when RENDERED_PAD_ROWS is
# 0, which links or copies the file untouched.
RENDERED_JPEG_QUALITY = 95

# Rate the frames under <clip>/frames were extracted at (process_ytb.py's --hz,
# set by scripts/vis3d/vis3d.sh, which runs the pipeline at 10 Hz). Everything
# under data/CARE_YTB is at 10 Hz: measured 155 frames over 15.5 s on ambulance,
# 437 over 43.7 s on four_way, one ego_poses.txt row per frame in both.
#
# This is the *input* rate. NAVSIM_HZ below is what comes out, and subsample()
# keeps every stride-th frame to get there -- at 10 Hz that is every 5th, so a
# 437-frame clip exports 88 scenes rather than 437.
#
# Only the sampling uses this. Timestamps come from NAVSIM_HZ via
# FRAME_INTERVAL_US and count the frames that were kept, while
# get_ego_dynamic_state() differences poses across *source* frames at
# 1/SOURCE_HZ, so speed is measured at the rate it was recorded at rather than
# at the rate it is exported at. Both stay right when this changes; what does
# not is leaving it at 2.0 while the frames are 10 Hz, which exports every
# frame as if it were 500 ms apart -- a five-fold error in every trajectory
# label, and nothing downstream would report it.
SOURCE_HZ = 10.0

# navsim frames are 2 Hz: NAVSIM_INTERVAL_LENGTH is 0.5 s and is baked into
# every Trajectory get_future_trajectory returns, so frames are dropped to
# match. Exporting at the source rate does not fail, it silently mislabels
# time -- a 10-frame future would be 1 s of footage that every consumer treats
# as 5 s, and the trajectory targets would be off by 5x.
NAVSIM_HZ = 2.0

# Which interleavings of the subsample to keep. Each phase is an equally valid
# 2 Hz sequence of the same clip, exported under its own log name, so listing
# more than one multiplies the scene count -- at the cost of scenes that
# overlap in content. (0,) keeps only the aligned one.
PHASES = (0,)

# False = copy each image into the tree; True = symlink it back to the clip.
#
# Copying, because the export is a dataset in its own right and a tree of links
# is only a view of one. A link dies if the clip directory is moved, renamed or
# re-rendered, and it dies quietly: navsim reads an unopenable image as
# np.zeros((1080,1920,3)) behind a bare except (dataclasses.py:79-82), so a
# broken tree trains on black frames and reports nothing. It also cannot be
# copied to another machine, which a link-free tree can.
#
# The cost is ~1.2 GB for the 2530 exported frames at ~457 KB each -- the real
# images only; the rendered tree is re-encoded for RENDERED_PAD_ROWS and has
# always been real files. Set True to go back to links if space is short.
LINK_SENSORS = False

# Skip a clip that has no ego_poses.txt. On by default because the pose fields
# *are* the training label: Scene.get_future_trajectory is built from nothing
# else, so a log without them trains on a label that says the car never moved.
# Set False only to export for rasterization/inspection, never for training.
REQUIRE_EGO_POSES = True

# Re-level the ego frame per clip before writing it.
#
# lift_frames_to_3d._camera_to_ego builds the ego frame as z = CAMERA_HEIGHT -
# y_cam: a pure axis swap plus a fixed lift, with no road-plane fit anywhere
# behind it. That is only the ego frame if the camera's optical axis is
# horizontal, and a dashcam angled down at the tarmac is not. The tilt shows up
# in the exported boxes as an underside that climbs linearly with range --
# measured at +6.3 deg on blocker and +5.7 deg on ice_road, which is a metre of
# false height every ten metres -- and buick_nearmiss shows the other half of
# the same gap, level but floating a constant 1.4 m.
#
# So the plane is fitted from the boxes themselves (they mostly do sit on the
# road) and the whole frame is rotated and shifted to put it at z = 0. It is a
# rigid change of basis, applied to the boxes, the ground geometry, the camera
# extrinsics and the pose track alike, so the log stays internally consistent
# and reprojection is unaffected. What it deliberately does NOT do is rescale
# anything: see the camera-height diagnostic it prints, which measures the
# depth-scale error left over rather than tuning it away.
CORRECT_ROAD_PLANE = True

# Classes whose underside is evidence about where the road is. Cars sit on it;
# a traffic light does not.
ROAD_PLANE_CLASSES = frozenset({"car", "truck", "bus", "van"})

# Below this many usable boxes the fit is noise and the clip is left alone.
ROAD_PLANE_MIN_BOXES = 40

# Refuse to rotate further than this. A clip that fits past it is not a
# mispointed camera, it is a broken lift, and silently rotating the frame would
# bury that rather than surface it.
ROAD_PLANE_MAX_PITCH_DEG = 15.0

# Theil-Sen is O(n^2) in pairs; more than this many boxes are sampled down to
# it. The estimate is already saturated well before this point.
ROAD_PLANE_MAX_SAMPLES = 1200

# What a dashcam is actually mounted at, for the diagnostic only. Nothing is
# scaled to make the corrected height match it.
NOMINAL_CAMERA_HEIGHT = 1.5

# Print the frames instead of writing them: no directory is created, no pickle
# written, no sensor linked or copied, nothing collected. Everything up to that
# point still runs, so a dry run reads the same inputs and fails on the same
# missing files as the real export. `python generate_ras_logs.py --dry-run`
# turns it on without editing this file.
DRY_RUN = False

# How much a dry run prints. The first DRY_RUN_FRAMES frames of each log are
# dumped field by field; every frame after that collapses to one line, so a
# 400-frame clip stays readable. Arrays and lists print at most
# DRY_RUN_ARRAY_ITEMS elements, and no single value more than one short line.
DRY_RUN_FRAMES = 2
DRY_RUN_ARRAY_ITEMS = 12
DRY_RUN_VALUE_CHARS = 160

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_CHANNEL = "CAM_F0"

FRAME_INTERVAL_US = int(1e6 / NAVSIM_HZ)

# Arbitrary but plausible epoch, so timestamps look like the real logs' rather
# than starting at zero. Nothing reads them for timing (see NAVSIM_HZ above).
TIMESTAMP_EPOCH_US = 1_620_000_000_000_000

# One-hot [left, right, straight, follow_lane]. A dashcam clip carries no
# route, so the command is follow-lane -- the same value the CARLA logs use
# when there is no route to condition on.
DRIVING_COMMAND = np.array([0, 0, 0, 1], dtype=np.int64)

# nuPlan's lidar sits behind and above the rear axle. For a clip the "lidar"
# frame is just the ego frame lift_frames_to_3d.py works in, so this is
# identity. Kept because the schema has the fields and consumers index them.
LIDAR2EGO_ROTATION = np.array([1.0, 0.0, 0.0, 0.0])
LIDAR2EGO_TRANSLATION = np.zeros(3)

# Camera (x right, y down, z forward) -> ego (x forward, y left, z up). The
# same remap as lift_frames_to_3d._camera_to_ego, as a matrix so it applies to
# a rotation by conjugation as well as to a translation.
CAMERA_TO_EGO_AXES = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])

# GroundingDINO's vocabulary -> navsim's. The classes navsim knows are
# vehicle, pedestrian, bicycle, traffic_cone, barrier and generic_object;
# anything unmapped becomes generic_object rather than being dropped, so a box
# that was detected stays visible in the raster even if its label is vague.
CLASS_MAP = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "motorcycle": "bicycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
}
DEFAULT_CLASS = "generic_object"

# ---------------------------------------------------------------------------
# Per-frame context
# ---------------------------------------------------------------------------
# Filled by build_frame() before it assembles the dict, so every get_<field>()
# below can be called with no arguments and the dict literal stays readable.
# Single-threaded by construction: build_frame is the only writer and it runs
# to completion before the next frame starts.
FRAME = {}


# ---------------------------------------------------------------------------
# One function per key of the frame dict
# ---------------------------------------------------------------------------

def get_anns() -> dict:
    """Detections in the ego frame.

    boxes_3d.json rows are already [x, y, z, l, w, h, heading] in the ego
    frame, which is navsim's own convention (see visualization/raster.py's
    _boxes_ego_to_world), so the geometry passes through untouched.
    """
    entry = FRAME["entry"]
    boxes = box_schema.to_array(entry.get("boxes", []))
    names = [CLASS_MAP.get(name, DEFAULT_CLASS) for name in entry.get("names", [])]
    count = len(boxes)
    return {
        "gt_boxes": boxes,
        "gt_names": np.asarray(names, dtype="<U14"),
        # Boxes are fit per frame and associated only within smooth_boxes.py,
        # whose track ids are not carried in boxes_3d.json, so there is no
        # measured velocity. Zeros are the honest value.
        "gt_velocity_3d": np.zeros((count, 3), dtype=np.float64),
        # ... and for the same reason an "instance" exists for one frame only:
        # its token cannot be shared with the same object in the next frame.
        "instance_tokens": [make_token("instance", FRAME["name"], str(i)) for i in range(count)],
        "track_tokens": [make_token("track", FRAME["name"], str(i)) for i in range(count)],
    }


def get_cams() -> dict:
    """The single front camera, from boxes_3d.json's per-frame camera entry.

    sensor2lidar_* is the inverse of the ego->camera transform the boxes were
    fit through: navsim stores the camera's pose in the ego/lidar frame, the
    opposite direction from what lift_frames_to_3d.py records.
    """
    camera = FRAME["camera"]
    ego_to_camera = np.asarray(camera["ego_to_camera"], dtype=np.float64)
    camera_to_ego = np.linalg.inv(ego_to_camera)
    return {
        CAMERA_CHANNEL: {
            # Blobs are named by frame token, not by the source frame number:
            # that is what the CARLA tree does, and it keeps the name unique
            # per log even though every clip counts frames from 000000.
            "data_path": f"{FRAME['log_name']}/{CAMERA_CHANNEL}/{FRAME['blob_name']}",
            "cam_intrinsic": np.asarray(camera["intrinsics"], dtype=np.float64),
            # UniDepth predicts a pinhole camera, so there is no distortion
            # model to carry: the point map, and the intrinsics recovered from
            # it, are already consistent with an undistorted pinhole.
            "distortion": np.zeros(5, dtype=np.float64),
            "sensor2lidar_rotation": camera_to_ego[:3, :3],
            "sensor2lidar_translation": camera_to_ego[:3, 3],
        }
    }


def get_can_bus() -> np.ndarray:
    """The 18-element can bus: [pos(3), quat(4), accel(3), vel(3), rate(3), 0, 0].

    Zeroed rather than filled. Nothing in navsim reads this field -- the ego
    state comes from ego_dynamic_state and ego2global_* -- so writing a
    derived-but-unverified copy of that information here would only create a
    second version to disagree with. (Note the layout above: the CARLA
    converter's docstring has velocity at [7:10], but the real trainval logs
    put acceleration there and velocity at [10:13], matching
    ego_dynamic_state's [vx, vy, ax, ay].)
    """
    return np.zeros(18, dtype=np.float64)


def get_driving_command() -> np.ndarray:
    return DRIVING_COMMAND.copy()


def get_ego2global() -> np.ndarray:
    return FRAME["pose"].copy()


def get_ego2global_rotation() -> np.ndarray:
    return quaternion_from_matrix(FRAME["pose"][:3, :3])


def get_ego2global_translation() -> np.ndarray:
    return FRAME["pose"][:3, 3].copy()


def get_ego_dynamic_state() -> list:
    """[vx, vy, ax, ay] in the ego frame, differenced from the pose track.

    Differenced across *source* frames, not the exported 2 Hz ones, so
    subsampling the clip does not change the measured speed.
    """
    poses, index = FRAME["poses"], FRAME["source_index"]
    if poses is None:
        return [0.0, 0.0, 0.0, 0.0]
    interval = 1.0 / SOURCE_HZ

    def velocity(i: int) -> np.ndarray:
        previous, following = max(i - 1, 0), min(i + 1, len(poses) - 1)
        if following == previous:
            return np.zeros(3)
        world = (poses[following, :3, 3] - poses[previous, :3, 3]) / ((following - previous) * interval)
        return poses[i, :3, :3].T @ world  # world -> this frame's own axes

    current = velocity(index)
    if index + 1 < len(poses):
        acceleration = (velocity(index + 1) - current) / interval
    elif index > 0:
        acceleration = (current - velocity(index - 1)) / interval
    else:
        acceleration = np.zeros(3)
    return [float(current[0]), float(current[1]),
            float(acceleration[0]), float(acceleration[1])]


def get_flow_gt_final_path():
    return None


def get_frame_idx() -> int:
    return FRAME["index"]


def get_lidar2ego() -> np.ndarray:
    return np.eye(4)


def get_lidar2ego_rotation() -> np.ndarray:
    return LIDAR2EGO_ROTATION.copy()


def get_lidar2ego_translation() -> np.ndarray:
    return LIDAR2EGO_TRANSLATION.copy()


def get_lidar2global() -> np.ndarray:
    """lidar2ego is identity here, so this is just the ego pose."""
    return FRAME["pose"].copy()


def get_lidar_path():
    """None: there is no lidar. Lidar.from_paths only touches this when
    "lidar_pc" is in the sensor config, which the drivoR agent leaves empty."""
    return None


def get_log_name() -> str:
    return FRAME["log_name"]


def get_log_token() -> str:
    return FRAME["log_token"]


def get_map_location() -> str:
    """The clip's own name, which is not a nuPlan map location -- which is
    exactly the point: Scene._build_map_api returns None for a name it does not
    recognise instead of failing to load a map that does not exist."""
    return FRAME["log_name"]


def get_occ_gt_final_path():
    return None


def get_roadblock_ids() -> list:
    """Empty: roadblocks are a map lookup. Scenes with no route only load under
    a scene filter with has_route: false."""
    return []


def get_sample_next():
    return FRAME["next_token"]


def get_sample_prev():
    return FRAME["prev_token"]


def get_scene_name() -> str:
    return FRAME["log_name"]


def get_scene_token() -> str:
    return FRAME["scene_token"]


def get_timestamp() -> int:
    return TIMESTAMP_EPOCH_US + FRAME["index"] * FRAME_INTERVAL_US


def get_token() -> str:
    return FRAME["token"]


def get_traffic_lights() -> list:
    """Empty. navsim's traffic_lights is a list of (lane_connector_id, is_red)
    -- a map reference resolved to a position at render time -- which cannot
    express a detected light's measured position or a three-way state. The
    measured ones go in traffic_lights_3d."""
    return []


def get_vehicle_name() -> str:
    return "dashcam"


# --- vis3d extras: ignored by stock navsim, read by raster_frames.py ---

def get_has_ego_pose() -> bool:
    return FRAME["poses"] is not None


def get_lanes() -> list:
    return FRAME["entry"].get("lanes", [])


def get_traffic_lights_3d() -> list:
    return FRAME["entry"].get("traffic_lights", [])


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def build_frame(context: dict) -> dict:
    """One navsim frame dict. `context` is what the get_<field>()s read."""
    global FRAME
    FRAME = context
    return {
        "anns": get_anns(),
        "cams": get_cams(),
        "can_bus": get_can_bus(),
        "driving_command": get_driving_command(),
        "ego2global": get_ego2global(),
        "ego2global_rotation": get_ego2global_rotation(),
        "ego2global_translation": get_ego2global_translation(),
        "ego_dynamic_state": get_ego_dynamic_state(),
        "flow_gt_final_path": get_flow_gt_final_path(),
        "frame_idx": get_frame_idx(),
        "lidar2ego": get_lidar2ego(),
        "lidar2ego_rotation": get_lidar2ego_rotation(),
        "lidar2ego_translation": get_lidar2ego_translation(),
        "lidar2global": get_lidar2global(),
        "lidar_path": get_lidar_path(),
        "log_name": get_log_name(),
        "log_token": get_log_token(),
        "map_location": get_map_location(),
        "occ_gt_final_path": get_occ_gt_final_path(),
        "roadblock_ids": get_roadblock_ids(),
        "sample_next": get_sample_next(),
        "sample_prev": get_sample_prev(),
        "scene_name": get_scene_name(),
        "scene_token": get_scene_token(),
        "timestamp": get_timestamp(),
        "token": get_token(),
        "traffic_lights": get_traffic_lights(),
        "vehicle_name": get_vehicle_name(),

        # --- extras, outside navsim's schema ---
        "has_ego_pose": get_has_ego_pose(),
        "lanes": get_lanes(),
        "traffic_lights_3d": get_traffic_lights_3d(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_token(*parts: str) -> str:
    """A stable 16-hex-character id, the shape navsim tokens have.

    Derived from the inputs rather than randomly, so re-exporting a clip
    produces the same tokens and anything keyed on them -- a metric cache, a
    scene filter's token list -- stays valid across re-runs.
    """
    return hashlib.md5("/".join(parts).encode()).hexdigest()[:16]


def quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a 3x3 rotation, the order navsim's Quaternion(*...) wants.

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


def load_ego_poses(poses_path: Path) -> np.ndarray:
    """(N, 4, 4) ego-to-global transforms from estimate_ego_motion.py's output.

    KITTI format: one line per frame, 12 numbers, a row-major 3x4
    camera-to-world matrix, starting from identity. Line i is source frame i.

    The poses are in camera axes and the ego convention is a fixed remap of
    those, so both the translation and (by conjugation) the rotation are
    converted here.
    """
    rows = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = CAMERA_TO_EGO_AXES @ rows[:, :3, :3] @ CAMERA_TO_EGO_AXES.T
    poses[:, :3, 3] = rows[:, :3, 3] @ CAMERA_TO_EGO_AXES.T
    return poses


def subsample(frame_names: list, phase: int) -> list:
    """Drops frames from SOURCE_HZ down to NAVSIM_HZ, keeping the phase-th of
    each `stride` (see NAVSIM_HZ and PHASES)."""
    stride = int(round(SOURCE_HZ / NAVSIM_HZ))
    if stride < 1:
        raise ValueError(f"SOURCE_HZ {SOURCE_HZ} is below navsim's {NAVSIM_HZ} Hz; "
                         "frames cannot be invented, re-extract the clip faster.")
    if not 0 <= phase < stride:
        raise ValueError(f"PHASES entries must be in [0, {stride}) for SOURCE_HZ={SOURCE_HZ}")
    return frame_names[phase::stride]


def fill_cameras(boxes_by_frame: dict, frame_names: list, run_dir: Path) -> dict:
    """A camera entry for every frame, including the ones that lifted nothing.

    The camera is recovered per frame from UniDepth's point map, so a frame the
    lifter skipped has none. It varies little along a clip -- same lens, same
    prediction -- so the nearest neighbour's is a far better answer than
    dropping the frame, and it only affects the intrinsics written into the log,
    not the boxes (there are none) or the pose.
    """
    known = {name: entry["camera"] for name, entry in boxes_by_frame.items()
             if "camera" in entry}
    if not known:
        raise KeyError(
            f"{run_dir/'boxes_3d.json'} has no per-frame 'camera' entry at all. Re-run "
            "lift_frames_to_3d.py: without it there is no camera to write into the log.")

    cameras, last = {}, None
    for name in frame_names:                      # forward fill
        last = known.get(name, last)
        cameras[name] = last
    last = None
    for name in reversed(frame_names):            # ... then backward, for the head
        last = known.get(name, last)
        if cameras[name] is None:
            cameras[name] = last
    return cameras


def find_ego_poses(clip_dir: Path, run_dir: Path):
    """estimate_ego_motion.py writes into the run directory, but older runs put
    ego_poses.txt beside the frames; accept either."""
    for candidate in (run_dir / "ego_poses.txt", clip_dir / "ego_poses.txt"):
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Dry-run printing
# ---------------------------------------------------------------------------

def clip_text(text: str) -> str:
    """One value's rendering, cut to a length the terminal can hold."""
    if len(text) <= DRY_RUN_VALUE_CHARS:
        return text
    return text[:DRY_RUN_VALUE_CHARS] + f" ... ({len(text)} chars)"


def format_value(value) -> str:
    """A leaf of a frame dict as text: dtype and shape first for arrays, so a
    dry run shows the schema as well as the numbers."""
    if isinstance(value, np.ndarray):
        head = f"{value.dtype} {tuple(value.shape)} "
        if value.size == 0:
            return head + "[]"
        with np.printoptions(precision=3, suppress=True, linewidth=88,
                             threshold=DRY_RUN_ARRAY_ITEMS, edgeitems=2):
            return head + clip_text(np.array2string(value, separator=", "))
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        shown = [clip_text(repr(item)) for item in value[:DRY_RUN_ARRAY_ITEMS]]
        if len(value) > DRY_RUN_ARRAY_ITEMS:
            shown.append(f"... {len(value)} total")
        return "[" + ", ".join(shown) + "]"
    return clip_text(repr(value))


def print_mapping(mapping: dict, indent: int) -> None:
    pad = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            print_mapping(value, indent + 2)
            continue
        # Keep a wrapped array under its own key rather than at column zero.
        text = format_value(value).replace("\n", "\n" + pad + " " * (len(key) + 2))
        print(f"{pad}{key}: {text}")


def print_frame(frame: dict, index: int, total: int) -> None:
    """One frame on stdout, in place of appending it to the pickle."""
    header = f"    [{index + 1}/{total}] {frame['cams'][CAMERA_CHANNEL]['data_path']}"
    if index >= DRY_RUN_FRAMES:
        x, y = frame["ego2global_translation"][:2]
        print(f"{header}  {frame['token']}  "
              f"{len(frame['anns']['gt_boxes'])} boxes, {len(frame['lanes'])} lanes, "
              f"{len(frame['traffic_lights_3d'])} lights, ego ({x:.2f}, {y:.2f})")
        return
    print(header)
    print_mapping(frame, 6)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def expand_clip_runs(clip_runs: list) -> list:
    """Resolves each "*" clip in CLIP_RUNS into the dataset's clip directories.

    A directory counts as a clip when the named run holds a boxes_3d.json --
    the one file every other input is read alongside. A dataset is normally
    part-way through being rendered, so a clip that has been downloaded but not
    lifted, or lifted into some other run, is reported and left out rather than
    written as a log with no boxes in it.

    Log names come from the clip name alone, so two datasets holding a clip of
    the same name would write one log over the other. Nothing downstream could
    tell that had happened -- the tree would simply be one scene short -- so it
    is refused here instead.
    """
    expanded = []
    for dataset, clip, run in clip_runs:
        if clip != "*":
            expanded.append((dataset, clip, run))
            continue
        dataset_dir = BASE / "data" / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"CLIP_RUNS names dataset {dataset!r}: no {dataset_dir}")
        found, skipped = [], []
        for path in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            run_dir = path / run if run else path
            (found if (run_dir / "boxes_3d.json").exists() else skipped).append(path.name)
        print(f"[{dataset}/*{'/' + run if run else ''}] {len(found)} clip(s): "
              f"{', '.join(found) if found else 'none'}")
        if skipped:
            print(f"  not rasterized at run {run!r}, skipping: {', '.join(skipped)}")
        expanded.extend((dataset, clip_name, run) for clip_name in found)

    seen = {}
    for dataset, clip, run in expanded:
        if clip in seen:
            raise ValueError(
                f"{seen[clip]}/{clip} and {dataset}/{clip} would both export as log "
                f"{clip!r}, and the second would overwrite the first.")
        seen[clip] = dataset
    return expanded


# ---------------------------------------------------------------------------
# Road plane
# ---------------------------------------------------------------------------

def fit_road_plane(boxes_by_frame: dict) -> tuple:
    """(pitch_rad, offset_m, n) of the plane the clip's boxes are sitting on.

    A box resting on the road has underside z - h/2 = 0, so the undersides of
    every vehicle in the clip are samples of the road surface as this log sees
    it, and a line through them against range is that surface in profile. If
    the ego frame were level and correctly placed the line would be y = 0; the
    slope it actually has is the pitch nobody modelled, and the intercept is
    the height offset.

    Fitted with Theil-Sen rather than least squares because the samples are
    contaminated by construction: some of those boxes are floating for reasons
    of their own -- a mis-lifted roof, a box smeared along the viewing ray --
    and a single 20 m outlier drags an ordinary regression off the road. The
    median of pairwise slopes ignores up to 29% of them.
    """
    distances, undersides = [], []
    for entry in boxes_by_frame.values():
        for box, name in zip(entry.get("boxes", []), entry.get("names", [])):
            if name not in ROAD_PLANE_CLASSES:
                continue
            row = box_schema.from_any(box)
            x, y, z = row[box_schema.CENTER]
            height = float(row[box_schema.HEIGHT])
            distances.append(np.hypot(x, y))
            undersides.append(z - height / 2.0)
    if len(distances) < ROAD_PLANE_MIN_BOXES:
        return None, None, len(distances)

    distance = np.asarray(distances)
    underside = np.asarray(undersides)
    if len(distance) > ROAD_PLANE_MAX_SAMPLES:
        pick = np.linspace(0, len(distance) - 1, ROAD_PLANE_MAX_SAMPLES).astype(int)
        distance, underside = distance[pick], underside[pick]

    left, right = np.triu_indices(len(distance), k=1)
    run_length = distance[right] - distance[left]
    # Pairs closer together in range than this have no lever arm on the slope:
    # their ratio is dominated by whatever noise is on the two undersides.
    usable = np.abs(run_length) > 1.0
    if not usable.any():
        return None, None, len(distances)
    slope = float(np.median((underside[right][usable] - underside[left][usable])
                            / run_length[usable]))
    intercept = float(np.median(underside - slope * distance))
    return float(np.arctan(slope)), intercept, len(distances)


def road_plane_transform(pitch: float, offset: float) -> np.ndarray:
    """The 4x4 that rotates the fitted plane level and drops it onto z = 0.

    A rotation about the ego y axis (left), which is pitch, then a lift. Taking
    a point on the fitted plane (d, y, d*tan(pitch) + offset) through it gives
    z = offset*cos(pitch) - offset*cos(pitch) = 0 for every d, which is the
    whole intent.
    """
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    transform = np.eye(4)
    transform[:3, :3] = [[cos_pitch, 0.0, sin_pitch],
                         [0.0, 1.0, 0.0],
                         [-sin_pitch, 0.0, cos_pitch]]
    transform[2, 3] = -offset * cos_pitch
    return transform


def _apply_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _reproject_ground(points, transform: np.ndarray, camera_to_ego: np.ndarray):
    """Move ground-plane geometry to where the corrected plane puts it.

    Lane centrelines are not measurements in space, they are image detections
    cast onto whatever the pipeline believed the road was -- hence their
    exactly-zero z and their tidy 1 m spacing in x. Rotating them like a box
    would tilt them off the road they are defined to lie on, and leaving them
    alone would keep them where the wrong plane put them. Both are wrong in the
    same way: the ray is the real observation, so the ray is what is kept. Each
    point is re-cast from the corrected camera centre and re-intersected with
    the corrected z = 0.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin = camera_to_ego[:3, 3]
    moved_origin = transform[:3, :3] @ origin + transform[:3, 3]
    moved = _apply_to_points(points, transform)

    direction = moved - moved_origin
    out = moved.copy()
    # A ray running parallel to the road never meets it; those points keep
    # their rotated position rather than being flung to the horizon.
    travelling = np.abs(direction[:, 2]) > 1e-9
    step = -moved_origin[2] / np.where(travelling, direction[:, 2], 1.0)
    forward = travelling & (step > 0)
    out[forward] = moved_origin + direction[forward] * step[forward, None]
    return out


def apply_road_plane(boxes_by_frame: dict, cameras: dict, poses, transform: np.ndarray):
    """Re-express every part of the clip in the corrected frame, in place.

    Everything the log states in ego coordinates has to move together or the
    log contradicts itself -- a box corrected without its camera reprojects
    into the wrong pixels, a box corrected without the pose track drifts
    against its own trajectory.
    """
    rotation = transform[:3, :3]
    inverse = np.linalg.inv(transform)

    for name, entry in boxes_by_frame.items():
        camera_to_ego = np.linalg.inv(np.asarray(
            cameras[name]["ego_to_camera"], dtype=np.float64))

        boxes = box_schema.to_array(entry.get("boxes", []))
        if len(boxes):
            boxes[:, box_schema.CENTER] = _apply_to_points(
                boxes[:, box_schema.CENTER], transform)
            # The heading is the box's long axis projected back onto the road.
            # Dimensions are left alone: this is a rigid motion, so the object
            # is the same size. Roll and pitch pass through untouched -- the
            # rows now carry them, but this correction is about levelling the
            # road plane, and re-deriving an object's own tilt from a rotation
            # of the ground is a different estimate than the one stored here.
            heading = boxes[:, box_schema.YAW]
            axis = np.stack([np.cos(heading), np.sin(heading), np.zeros_like(heading)], 1)
            axis = axis @ rotation.T
            boxes[:, box_schema.YAW] = np.arctan2(axis[:, 1], axis[:, 0])
            entry["boxes"] = box_schema.to_dicts(boxes)

        if entry.get("lanes"):
            entry["lanes"] = [_reproject_ground(lane, transform, camera_to_ego).tolist()
                              for lane in entry["lanes"] if len(lane)]

        for light in entry.get("traffic_lights", []):
            if isinstance(light, dict) and "position" in light:
                light["position"] = _apply_to_points(
                    [light["position"]], transform).reshape(3).tolist()

        # camera_to_ego_new = T @ camera_to_ego_old, stored in the direction
        # boxes_3d.json keeps it.
        cameras[name]["ego_to_camera"] = (
            np.asarray(cameras[name]["ego_to_camera"], dtype=np.float64) @ inverse).tolist()

    if poses is not None:
        # A change of basis on a frame that is itself moving: the pose maps ego
        # to global and both ends are re-expressed, so it conjugates. Frame 0
        # stays identity, which keeps the trajectory label anchored where every
        # consumer expects it.
        for index in range(len(poses)):
            poses[index] = transform @ poses[index] @ inverse


def report_road_plane(log_name: str, pitch: float, offset: float, count: int,
                      cameras: dict, boxes_by_frame: dict) -> None:
    """Say what was corrected, and measure what was left behind.

    Grounding the boxes and keeping the camera at its nominal height are two
    different claims about the same scene, and depth scale is what reconciles
    them. Correcting the plane picks the first, which leaves the camera at
    whatever height the depth field implies -- so that height, divided by the
    1.5 m the lift assumed, is a direct read of the scale error this correction
    does not touch. It is printed rather than applied.
    """
    height = float(np.linalg.inv(np.asarray(
        next(iter(cameras.values()))["ego_to_camera"], dtype=np.float64))[2, 3])
    print(f"  road plane: pitch {np.degrees(pitch):+.1f} deg, offset {offset:+.2f} m "
          f"(fitted on {count} boxes)")
    print(f"  camera now sits at {height:.2f} m; nominal is {NOMINAL_CAMERA_HEIGHT:.2f} m "
          f"=> depth scale is off by ~{height / NOMINAL_CAMERA_HEIGHT:.2f}x, NOT corrected here")

    # Near/far size split. A clip whose close boxes and distant boxes disagree
    # about how big a car is has a depth field no rigid correction can save,
    # and the export should say so out loud rather than leave it to an audit.
    near, far = [], []
    for entry in boxes_by_frame.values():
        for box, name in zip(entry.get("boxes", []), entry.get("names", [])):
            if name != "car":
                continue
            row = box_schema.from_any(box)
            distance = np.hypot(float(row[box_schema.X]), float(row[box_schema.Y]))
            (near if distance < 25.0 else far).append(float(row[box_schema.LENGTH]))
    if len(near) >= 10 and len(far) >= 10:
        near_median, far_median = float(np.median(near)), float(np.median(far))
        ratio = near_median / far_median if far_median else float("inf")
        if ratio > 1.6 or ratio < 0.625:
            print(f"  WARNING {log_name}: car length is {near_median:.1f} m inside 25 m "
                  f"but {far_median:.1f} m beyond it ({ratio:.1f}x). The depth field is "
                  "range-dependent; no rigid correction and no single scale factor fixes "
                  "that. These boxes are not usable as size labels until the lift is redone.")


def place_blob(source: Path, destination: Path) -> None:
    """Put one image into a blob tree, replacing whatever is already there."""
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if LINK_SENSORS:
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def place_rendered_blob(source: Path, destination: Path) -> None:
    """Same, but grown to the height navsim's [20:-20] crop expects.

    See RENDERED_PAD_ROWS for why the margin is needed. Adding it forces a
    decode and re-encode, so the rendered tree cannot be symlinked the way the
    real one is; with the padding off this is just place_blob.
    """
    if not RENDERED_PAD_ROWS:
        place_blob(source, destination)
        return

    # Imported here rather than at module scope so the dependency lands only on
    # runs that actually pad: a dry run, or an export with RENDERED_PAD_ROWS 0,
    # still needs nothing but numpy and the stdlib.
    from PIL import Image

    if destination.exists() or destination.is_symlink():
        destination.unlink()
    with Image.open(source) as opened:
        # The destination name carries the real frame's suffix, so a PNG render
        # can end up being written as JPEG, which has no alpha channel.
        image = (opened.convert("RGB")
                 if destination.suffix.lower() in {".jpg", ".jpeg"} else opened)
        padded = Image.new(image.mode,
                           (image.width, image.height + 2 * RENDERED_PAD_ROWS))
        padded.paste(image, (0, RENDERED_PAD_ROWS))
        padded.save(destination, quality=RENDERED_JPEG_QUALITY)


def export_clip(dataset: str, clip: str, run: str, phase: int) -> None:
    clip_dir = BASE / "data" / dataset / clip
    run_dir = clip_dir / run if run else clip_dir
    frames_dir = clip_dir / "frames"

    def source_dir(source: str) -> Path:
        return frames_dir if source == "frames" else run_dir / source

    real_dir = source_dir(REAL_SENSOR_SOURCE)
    rendered_dir = source_dir(RENDERED_SENSOR_SOURCE)

    stride = int(round(SOURCE_HZ / NAVSIM_HZ))
    log_name = clip if stride == 1 or phase == 0 else f"{clip}_phase{phase}"

    with open(run_dir / "boxes_3d.json") as file:
        boxes_by_frame = json.load(file)

    # boxes_3d.json carries top-level metadata alongside the frames, keyed with a
    # leading underscore: "_manual_3d" from apply_manual_boxes_3d.py and
    # "_ego_filtered" from drop_ego_artifacts.py. They are flags, not frames, and
    # four functions below iterate this dict expecting every value to be a frame
    # entry -- fill_cameras crashes on the first one with "argument of type 'bool'
    # is not iterable". Dropped once here rather than guarded in each of them, so
    # a fifth reader added later cannot miss the convention.
    boxes_by_frame = {name: entry for name, entry in boxes_by_frame.items()
                      if not name.startswith("_")}

    # Driven by the images, not by boxes_3d.json's keys. boxes_3d.json only
    # carries a frame something lifted in -- 318 of snowcrash's 407 -- and a
    # log that skips the rest is not merely sparse, it is *wrong*: navsim reads
    # consecutive log frames as NAVSIM_INTERVAL_LENGTH apart, so a hole silently
    # stretches the gap it sits in and every trajectory label spanning it is
    # off. A frame with no detections is a real observation of an empty road
    # and belongs in the log with empty anns.
    def list_images(directory: Path, setting: str, source: str) -> list:
        names = sorted(path.name for path in directory.iterdir()
                       if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not names:
            raise FileNotFoundError(
                f"No images in {directory}. Check {setting} ({source!r}) and "
                "that the run has been rasterized.")
        return names

    frame_names = list_images(real_dir, "REAL_SENSOR_SOURCE", REAL_SENSOR_SOURCE)
    rendered_names = set(
        list_images(rendered_dir, "RENDERED_SENSOR_SOURCE", RENDERED_SENSOR_SOURCE))

    # Checked here because it is not checked anywhere later: a frame that exists
    # in one tree and not the other is read as np.zeros and trains as a black
    # image, so an export that is short a rasterization has to fail now.
    unrendered = [name for name in frame_names if name not in rendered_names]
    if unrendered:
        raise FileNotFoundError(
            f"{len(unrendered)} of {len(frame_names)} frames under {real_dir} have "
            f"no rasterization in {rendered_dir} (first: {unrendered[0]}). Both "
            "trees must cover the same frames; re-run the rasterize stage.")
    cameras = fill_cameras(boxes_by_frame, frame_names, run_dir)
    empty = sum(1 for name in frame_names if name not in boxes_by_frame)
    if empty:
        print(f"  {empty} of {len(frame_names)} frames lifted nothing; "
              "written with empty annotations")

    poses_path = find_ego_poses(clip_dir, run_dir)
    if poses_path is None:
        message = (f"  {clip}: no ego_poses.txt in {run_dir} or {clip_dir} -- the trajectory "
                   "label would say the car never moves. Run vis3d/estimate_ego_motion.py.")
        if REQUIRE_EGO_POSES:
            print(message + " Skipping.")
            return
        print(message + " Exporting anyway (REQUIRE_EGO_POSES is off): NOT for training.")
    poses = load_ego_poses(poses_path) if poses_path else None

    if CORRECT_ROAD_PLANE:
        pitch, offset, count = fit_road_plane(boxes_by_frame)
        if pitch is None:
            print(f"  road plane: only {count} boxes of a road-going class "
                  f"(need {ROAD_PLANE_MIN_BOXES}); frame left as lifted")
        elif abs(np.degrees(pitch)) > ROAD_PLANE_MAX_PITCH_DEG:
            print(f"  road plane: fitted pitch {np.degrees(pitch):+.1f} deg exceeds "
                  f"{ROAD_PLANE_MAX_PITCH_DEG:.0f} deg; refusing to rotate. The lift for "
                  "this clip is wrong in some way this correction would only hide.")
        else:
            apply_road_plane(boxes_by_frame, cameras, poses,
                             road_plane_transform(pitch, offset))
            report_road_plane(log_name, pitch, offset, count, cameras, boxes_by_frame)

    available = len(frame_names)
    frame_names = subsample(frame_names, phase)
    print(f"  {log_name}: {len(frame_names)} of {available} frames "
          f"({SOURCE_HZ:g} Hz -> {NAVSIM_HZ:g} Hz, phase {phase})")

    if poses is not None:
        needed = max(int(Path(name).stem) for name in frame_names) + 1
        if len(poses) < needed:
            raise ValueError(
                f"{poses_path} has {len(poses)} poses but frame {needed - 1:06d} needs one. "
                "The pose file must come from the same video extracted at the same rate "
                "as the frames.")

    # One clip is one log and one scene, so these are the same token -- which is
    # also what the CARLA tree does: log_token, scene_token and the pickle's own
    # name are one value there, and the synthetic stub is keyed on it too.
    log_token = scene_token = make_token("log", log_name)
    tokens = [make_token(log_name, name) for name in frame_names]
    blob_names = [f"{token}{Path(name).suffix}" for token, name in zip(tokens, frame_names)]

    log_path = DATASET_ROOT / META_DIR_NAME / f"{scene_token}.pkl"
    blob_dir = DATASET_ROOT / BLOB_DIR_NAME / log_name / CAMERA_CHANNEL
    rendered_blob_dir = (DATASET_ROOT / RENDERED_BLOB_DIR_NAME
                         / log_name / CAMERA_CHANNEL)
    if not DRY_RUN:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        blob_dir.mkdir(parents=True, exist_ok=True)
        rendered_blob_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index, name in enumerate(frame_names):
        source_index = int(Path(name).stem)
        frame = build_frame({
            "entry": boxes_by_frame.get(name, {}),
            "camera": cameras[name],
            "name": name,
            "blob_name": blob_names[index],
            "index": index,
            "token": tokens[index],
            "prev_token": tokens[index - 1] if index > 0 else None,
            "next_token": tokens[index + 1] if index + 1 < len(frame_names) else None,
            "log_name": log_name,
            "log_token": log_token,
            "scene_token": scene_token,
            "poses": poses,
            "source_index": source_index,
            "pose": poses[source_index] if poses is not None else np.eye(4),
        })
        frames.append(frame)

        if DRY_RUN:
            print_frame(frame, index, len(frame_names))
            continue

        # One name, both trees: that shared name is the whole of the pairing
        # dataclasses.py:77 relies on.
        place_blob(real_dir / name, blob_dir / blob_names[index])
        place_rendered_blob(rendered_dir / name,
                            rendered_blob_dir / blob_names[index])

    if not DRY_RUN:
        with open(log_path, "wb") as file:
            pickle.dump(frames, file)

    boxes = sum(len(frame["anns"]["gt_boxes"]) for frame in frames)
    lanes = sum(len(frame["lanes"]) for frame in frames)
    lights = sum(len(frame["traffic_lights_3d"]) for frame in frames)
    prefix = "would write " if DRY_RUN else ""
    transfer = "symlink" if LINK_SENSORS else "copy"
    rendered_transfer = (f"pad {RENDERED_PAD_ROWS}px + write" if RENDERED_PAD_ROWS
                         else transfer)
    count = f" ({len(frame_names)} images)" if DRY_RUN else ""
    print(f"    {prefix}{log_path}  ({boxes} boxes, {lanes} lanes, {lights} lights)")
    print(f"    {'would ' + transfer + ' ' if DRY_RUN else ''}{blob_dir}"
          f"  <- {real_dir}{count}")
    print(f"    {'would ' + rendered_transfer + ' ' if DRY_RUN else ''}{rendered_blob_dir}"
          f"  <- {rendered_dir}{count}")
    write_synthetic_scene(log_name, scene_token, tokens[0])


def write_synthetic_scene(log_name: str, scene_token: str, initial_token: str) -> None:
    """The synthetic_scene_pickles entry for one clip, in the CARLA tree's shape.

    That shape is a stub: scene_metadata and three empty lists. The two-stage
    evaluation is the only thing that reads these, out of its own
    NAVHARD_DATA_ROOT, and it needs a per-frame reactive rollout that a dashcam
    clip has no way to produce -- so the lists stay empty and the file exists
    only to keep the three directories in step. Training never opens it.
    """
    if not WRITE_SYNTHETIC_SCENES:
        return
    scene = {
        "scene_metadata": {
            "log_name": log_name,
            "scene_token": scene_token,
            "map_name": log_name,      # same non-nuPlan name as map_location
            "initial_token": initial_token,
        },
        "frames": [],
        "extended_traffic_light_data": [],
        "extended_detections_tracks": [],
    }
    path = DATASET_ROOT / SYNTHETIC_DIR_NAME / f"{scene_token}.pkl"
    if DRY_RUN:
        print(f"    would write {path}  (stub)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(scene, file)
    print(f"    {path}  (stub)")


def ensure_dataset_tree() -> None:
    """Create DATASET_ROOT and its three subdirectories, empty.

    Every writer below already mkdirs its own parent, so on a normal run this
    changes nothing. It matters when the tree is being rebuilt from scratch:
    those mkdirs only fire for a directory something is actually written into,
    so a wiped root plus a skipped clip (no ego_poses.txt), WRITE_SYNTHETIC_
    SCENES off, or an export that dies partway leaves a half-built tree. The
    result is a root whose shape depends on what happened to succeed, and the
    missing directory does not surface until a training run points
    sim_log_path at it.

    Creating all three up front makes the layout a property of the script
    rather than of the run. exist_ok, so re-running over a populated tree is a
    no-op and nothing existing is touched or removed.
    """
    for name in (META_DIR_NAME, BLOB_DIR_NAME, RENDERED_BLOB_DIR_NAME,
                 SYNTHETIC_DIR_NAME):
        (DATASET_ROOT / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    global DRY_RUN
    unknown = [arg for arg in sys.argv[1:] if arg != "--dry-run"]
    if unknown:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} [--dry-run]"
                         f"  (unrecognised: {' '.join(unknown)})")
    DRY_RUN = DRY_RUN or "--dry-run" in sys.argv[1:]
    if DRY_RUN:
        print("DRY RUN: printing the frames instead of writing them. "
              "Nothing is created, written, linked or copied.")

    print(f"Writing to {DATASET_ROOT} (real: {REAL_SENSOR_SOURCE}, "
          f"rendered: {RENDERED_SENSOR_SOURCE}"
          f"{f' +{RENDERED_PAD_ROWS}px' if RENDERED_PAD_ROWS else ''})")
    print(f"  {META_DIR_NAME}/, {BLOB_DIR_NAME}/, {RENDERED_BLOB_DIR_NAME}/"
          f"{', ' + SYNTHETIC_DIR_NAME + '/' if WRITE_SYNTHETIC_SCENES else ''}")
    if not DRY_RUN:
        ensure_dataset_tree()
    for dataset, clip, run in expand_clip_runs(CLIP_RUNS):
        print(f"[{dataset}/{clip}{'/' + run if run else ''}]")
        for phase in PHASES:
            export_clip(dataset, clip, run, phase)

    # The two settings these logs will not load under, stated once where the
    # person running the export will see it.
    print("\nTo train on these:")
    print(f"  - sim_log_path: {DATASET_ROOT / META_DIR_NAME}")
    print(f"    sim_sensor_path: {DATASET_ROOT / BLOB_DIR_NAME}")
    print(f"    ({RENDERED_BLOB_DIR_NAME}/ is configured nowhere -- "
          "dataclasses.py:77 finds it by substituting into the path above)")
    print(f"  - only {CAMERA_CHANNEL} is written. rap_agent.py:150 asks for "
          "cam_f0/l0/r0/b0; the three a dashcam does not have load as zeros")
    print("    (run_training_full.py's sim path already forces has_route false "
          "and drops the log-name/token filters)")
    print("  - a scene needs num_history_frames + num_future_frames frames "
          f"(14 for navtrain) = {14 / NAVSIM_HZ:g} s of clip")
    print("  - tokens have no PDM metric cache, so compute_score returns zeros "
          "for them; mask them out of the score loss the way sim rows are")


if __name__ == "__main__":
    main()

"""
DRY RUN: printing the frames instead of writing them. Nothing is created, written, linked or copied.
Writing each clip's log to <run_dir>/navsim_logs (sensors: vis3d)
... and linking them into /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/navsim_export/navsim_logs/video
[beepbeep/1]
  beepbeep: 33 of 33 frames (2 Hz -> 2 Hz, phase 0)
    [1/33] beepbeep/CAM_F0/000000.jpg
      anns:
        gt_boxes: float64 (4, 7) [[  9.512, -23.104, ...,   1.635,   1.542],
                   [ 18.716, -36.901, ...,   1.664,  -1.076],
                   [ 18.74 , -55.265, ...,   0.742,  -1.571],
                   [ 13.525, -50.318, ...,   1 ... (175 chars)
        gt_names: <U14 (4,) ['vehicle', 'vehicle', 'vehicle', 'vehicle']
        gt_velocity_3d: float64 (4, 3) [[0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]]
        instance_tokens: ['bd48c76200ed6082', '9a621dfd58998e91', '99cc9478348dcf00', '22dbbaea693f8284']
        track_tokens: ['02c15584c7c9cb08', 'cdc831fb3990c574', '5b18d00064c26724', '82b36dcf86f6b3cb']
      cams:
        CAM_F0:
          data_path: 'beepbeep/CAM_F0/000000.jpg'
          cam_intrinsic: float64 (3, 3) [[258.443,   0.   , 307.5  ],
                          [  0.   , 258.443, 230.75 ],
                          [  0.   ,   0.   ,   1.   ]]
          distortion: float64 (5,) [0., 0., 0., 0., 0.]
          sensor2lidar_rotation: float64 (3, 3) [[ 0.,  0.,  1.],
                                  [-1., -0., -0.],
                                  [-0., -1., -0.]]
          sensor2lidar_translation: float64 (3,) [ 0. , -0. ,  1.5]
      can_bus: float64 (18,) [0., 0., ..., 0., 0.]
      driving_command: int64 (4,) [0, 0, 0, 1]
      ego2global: float64 (4, 4) [[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]]
      ego2global_rotation: float64 (4,) [1., 0., 0., 0.]
      ego2global_translation: float64 (3,) [0., 0., 0.]
      ego_dynamic_state: [0.3318859338760376, 0.030884675681591034, -0.8493032389867728, 0.03920799549780683]
      flow_gt_final_path: None
      frame_idx: 0
      lidar2ego: float64 (4, 4) [[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]
      lidar2ego_rotation: float64 (4,) [1., 0., 0., 0.]
      lidar2ego_translation: float64 (3,) [0., 0., 0.]
      lidar2global: float64 (4, 4) [[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]]
      lidar_path: None
      log_name: 'beepbeep'
      log_token: 'e169f73417451c4e'
      map_location: 'beepbeep'
      occ_gt_final_path: None
      roadblock_ids: []
      sample_next: '41327ae69fedee15'
      sample_prev: None
      scene_name: 'beepbeep'
      scene_token: '682d267c33bd1338'
      timestamp: 1620000000000000
      token: '0b729899bb522b0a'
      traffic_lights: []
      vehicle_name: 'dashcam'
      has_ego_pose: True
      lanes: []
      traffic_lights_3d: []
    [2/33] beepbeep/CAM_F0/000001.jpg
      anns:
        gt_boxes: float64 (5, 7) [[ 11.582, -27.306, ...,   1.941,   1.549],
                   [ 15.195, -28.274, ...,   1.211,  -1.047],
                   ...,
                   [ 16.107, -46.785, ...,   0.674,  -1.571],
                   [ 17.86 , -36.84 , .. ... (181 chars)
        gt_names: <U14 (5,) ['vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle']
        gt_velocity_3d: float64 (5, 3) [[0., 0., 0.],
                         [0., 0., 0.],
                         ...,
                         [0., 0., 0.],
                         [0., 0., 0.]]
        instance_tokens: ['7917536bcc8fb37f', '45557d3e81046af5', '3202d2a4daaa533b', '97d5f24a4422d731', '815d2ddca9f6b0b8']
        track_tokens: ['a3660726c65438f8', '8bd935925c7553da', 'aa45775bd93c86cb', '6d54482f711f6fcd', 'f704b4c41d7c3e55']
      cams:
        CAM_F0:
          data_path: 'beepbeep/CAM_F0/000001.jpg'
          cam_intrinsic: float64 (3, 3) [[258.443,   0.   , 307.5  ],
                          [  0.   , 258.443, 230.75 ],
                          [  0.   ,   0.   ,   1.   ]]
          distortion: float64 (5,) [0., 0., 0., 0., 0.]
          sensor2lidar_rotation: float64 (3, 3) [[ 0.,  0.,  1.],
                                  [-1., -0., -0.],
                                  [-0., -1., -0.]]
          sensor2lidar_translation: float64 (3,) [ 0. , -0. ,  1.5]
      can_bus: float64 (18,) [0., 0., ..., 0., 0.]
      driving_command: int64 (4,) [0, 0, 0, 1]
      ego2global: float64 (4, 4) [[ 0.999,  0.042,  0.031,  0.166],
                   [-0.04 ,  0.998, -0.039,  0.015],
                   [-0.033,  0.038,  0.999,  0.   ],
                   [ 0.   ,  0.   ,  0.   ,  1.   ]]
      ego2global_rotation: float64 (4,) [ 0.999,  0.019,  0.016, -0.021]
      ego2global_translation: float64 (3,) [0.166, 0.015, 0.   ]
      ego_dynamic_state: [-0.09276568561734881, 0.05048867343049445, -1.359585458730667, -0.008131775722976672]
      flow_gt_final_path: None
      frame_idx: 1
      lidar2ego: float64 (4, 4) [[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]]
      lidar2ego_rotation: float64 (4,) [1., 0., 0., 0.]
      lidar2ego_translation: float64 (3,) [0., 0., 0.]
      lidar2global: float64 (4, 4) [[ 0.999,  0.042,  0.031,  0.166],
                     [-0.04 ,  0.998, -0.039,  0.015],
                     [-0.033,  0.038,  0.999,  0.   ],
                     [ 0.   ,  0.   ,  0.   ,  1.   ]]
      lidar_path: None
      log_name: 'beepbeep'
      log_token: 'e169f73417451c4e'
      map_location: 'beepbeep'
      occ_gt_final_path: None
      roadblock_ids: []
      sample_next: '6995c755abbc3f0f'
      sample_prev: '0b729899bb522b0a'
      scene_name: 'beepbeep'
      scene_token: '682d267c33bd1338'
      timestamp: 1620000000500000
      token: '41327ae69fedee15'
      traffic_lights: []
      vehicle_name: 'dashcam'
      has_ego_pose: True
      lanes: []
      traffic_lights_3d: []
    [3/33] beepbeep/CAM_F0/000002.jpg  6995c755abbc3f0f  6 boxes, 0 lanes, 0 lights, ego (-0.09, 0.05)
    [4/33] beepbeep/CAM_F0/000003.jpg  480bae4e00205c66  6 boxes, 0 lanes, 0 lights, ego (-0.60, 0.11)
    [5/33] beepbeep/CAM_F0/000004.jpg  18eff3f2fc60ecbe  6 boxes, 0 lanes, 0 lights, ego (-1.17, 0.20)
    [6/33] beepbeep/CAM_F0/000005.jpg  143839192b44ef36  5 boxes, 0 lanes, 0 lights, ego (-1.70, 0.33)
    [7/33] beepbeep/CAM_F0/000006.jpg  25d587555a6f4824  6 boxes, 0 lanes, 0 lights, ego (-2.29, 0.46)
    [8/33] beepbeep/CAM_F0/000007.jpg  7891fe80543bc760  7 boxes, 0 lanes, 0 lights, ego (-2.99, 0.60)
    [9/33] beepbeep/CAM_F0/000008.jpg  314ad21e59a9e86c  8 boxes, 0 lanes, 0 lights, ego (-3.75, 0.73)
    [10/33] beepbeep/CAM_F0/000009.jpg  f8013854bff70d9a  9 boxes, 0 lanes, 0 lights, ego (-4.53, 0.88)
    [11/33] beepbeep/CAM_F0/000010.jpg  702170d9d4840812  9 boxes, 0 lanes, 0 lights, ego (-5.29, 1.05)
    [12/33] beepbeep/CAM_F0/000011.jpg  885548abf94e08f7  9 boxes, 0 lanes, 0 lights, ego (-6.03, 1.22)
    [13/33] beepbeep/CAM_F0/000012.jpg  7536b2f7558dff3c  9 boxes, 0 lanes, 0 lights, ego (-6.67, 1.40)
    [14/33] beepbeep/CAM_F0/000013.jpg  eb83573fe6c084cc  11 boxes, 0 lanes, 0 lights, ego (-7.23, 1.52)
    [15/33] beepbeep/CAM_F0/000014.jpg  96e3c7957d49287f  11 boxes, 0 lanes, 0 lights, ego (-7.69, 1.61)
    [16/33] beepbeep/CAM_F0/000015.jpg  00623746b33bfdb3  12 boxes, 0 lanes, 0 lights, ego (-8.22, 1.66)
    [17/33] beepbeep/CAM_F0/000016.jpg  01b765d051e6330b  11 boxes, 0 lanes, 0 lights, ego (-8.81, 1.67)
    [18/33] beepbeep/CAM_F0/000017.jpg  6bb47f4e89f794c2  12 boxes, 0 lanes, 0 lights, ego (-8.59, 1.71)
    [19/33] beepbeep/CAM_F0/000018.jpg  db0779fe4edeb7c1  12 boxes, 0 lanes, 0 lights, ego (-8.63, 1.75)
    [20/33] beepbeep/CAM_F0/000019.jpg  f6400a1736052fe9  11 boxes, 0 lanes, 0 lights, ego (-8.95, 1.76)
    [21/33] beepbeep/CAM_F0/000020.jpg  8b38efff2c825f33  9 boxes, 0 lanes, 0 lights, ego (-9.28, 1.77)
    [22/33] beepbeep/CAM_F0/000021.jpg  aa753d0b8c5e0f72  6 boxes, 0 lanes, 0 lights, ego (-9.68, 1.73)
    [23/33] beepbeep/CAM_F0/000022.jpg  8fd615c2eb14b897  7 boxes, 0 lanes, 0 lights, ego (-9.95, 1.64)
    [24/33] beepbeep/CAM_F0/000023.jpg  4c04d39ffc666fd6  10 boxes, 0 lanes, 0 lights, ego (-10.29, 1.47)
    [25/33] beepbeep/CAM_F0/000024.jpg  f1266c2e281b050b  9 boxes, 0 lanes, 0 lights, ego (-10.58, 1.35)
    [26/33] beepbeep/CAM_F0/000025.jpg  9286fc89f1d0136a  9 boxes, 0 lanes, 0 lights, ego (-10.86, 1.24)
    [27/33] beepbeep/CAM_F0/000026.jpg  2e48fba5ac797d30  9 boxes, 0 lanes, 0 lights, ego (-11.21, 1.08)
    [28/33] beepbeep/CAM_F0/000027.jpg  656e3e813b34b043  8 boxes, 0 lanes, 0 lights, ego (-11.60, 0.91)
    [29/33] beepbeep/CAM_F0/000028.jpg  8e6ada4c03f5ae75  10 boxes, 0 lanes, 0 lights, ego (-12.03, 0.70)
    [30/33] beepbeep/CAM_F0/000029.jpg  45ef6d1aa736f7e8  10 boxes, 0 lanes, 0 lights, ego (-12.49, 0.45)
    [31/33] beepbeep/CAM_F0/000030.jpg  64c700ccde4d6676  10 boxes, 0 lanes, 0 lights, ego (-12.98, 0.16)
    [32/33] beepbeep/CAM_F0/000031.jpg  6612865ca9296579  10 boxes, 0 lanes, 0 lights, ego (-13.47, -0.17)
    [33/33] beepbeep/CAM_F0/000032.jpg  561e8dc3d02727c3  10 boxes, 0 lanes, 0 lights, ego (-13.95, -0.53)
    would write /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/beepbeep/1/navsim_logs/beepbeep.pkl  (286 boxes, 0 lanes, 0 lights)
    would symlink /fs/nexus-projects/sim2real/aliu/RAP/CARE/sensor_blobs/beepbeep/CAM_F0  <- /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/beepbeep/frames (33 images)
    would pad 20px + write /fs/nexus-projects/sim2real/aliu/RAP/CARE/rendered_sensor_blobs/beepbeep/CAM_F0  <- /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/beepbeep/1/vis3d (33 images)
    not collected: /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/navsim_export/navsim_logs/video/beepbeep.pkl already exists and is not a link
    not collected: /fs/nexus-projects/sim2real/aliu/RAP/data/YTB/navsim_export/sensor_blobs/video/beepbeep already exists and is not a link

To train on these:
  - scene_filter.has_route must be false (roadblock_ids is empty)
  - a scene needs num_history_frames + num_future_frames frames (14 for navtrain) = 7 s of clip
  - tokens have no PDM metric cache, so compute_score returns zeros for them; mask them out of the score loss the way sim rows are
"""