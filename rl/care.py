"""CARE clips as scenes the RAP model and the scorer can both consume.

Why this is not `navsim.common.dataloader.SceneLoader`
-----------------------------------------------------
The CARE export is openscene-format -- `generate_ras_logs.py` writes exactly the
layout `SceneLoader` reads -- but two things in the devkit's loader stop on it, and
both are in `navsim/`, which this package does not edit:

  * `Scene.from_scene_dict_list` calls `_build_map_api(map_name)`, which asserts
    the name is one of the four nuPlan maps. A CARE clip's `map_location` is the
    clip's own name, so building a `Scene` -- the only route to the human
    trajectory and the annotations -- raises before anything is loaded.
  * `AgentInput.from_scene_dict_list` does `Path(frame["lidar_path"])`
    unconditionally, and the CARE frames carry `lidar_path: None` (there is no
    lidar, and the agent's sensor config asks for none). It raises on the `Path`
    call before the sensor config is ever consulted.

So this module walks the frame dicts itself. It is deliberately a re-implementation
of those two constructors and nothing more: same fields, same order, same relative
pose convention, so a CARE sample and a navtrain sample reach the model as the same
kind of thing.

What a window is
----------------
One clip pickle is a list of frames at 0.5 s. A *scene* is a window of
`num_history_frames + num_future_frames` of them; the frame at index
`num_history_frames - 1` is "now", the model sees the history up to and including
it, and the poses after it are the human trajectory. Windows step by
`care_frame_stride`, so a 74-frame clip yields either ~61 overlapping scenes
(stride 1) or 5 disjoint ones (stride 14).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pickle

import numpy as np
import torch


@dataclass
class CareScene:
    """One scored-able window of one clip."""

    token: str
    log_name: str
    clip_pickle: Path
    start_index: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CareScene({self.log_name}@{self.start_index}, {self.token[:8]})"


def _pose_of(frame: Dict) -> np.ndarray:
    """Global (x, y, yaw) of one frame's ego, the way navsim derives it."""
    from pyquaternion import Quaternion

    translation = frame["ego2global_translation"]
    quaternion = Quaternion(*frame["ego2global_rotation"])
    return np.array(
        [translation[0], translation[1], quaternion.yaw_pitch_roll[0]], dtype=np.float64
    )


def _relative_poses(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """`poses` expressed in the frame of `origin`. Same helper navsim uses, so the
    ego-frame convention is identical on both data paths."""
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array,
    )

    return convert_absolute_to_relative_se2_array(StateSE2(*origin), np.asarray(poses))


def agent_box_corners(boxes: np.ndarray) -> np.ndarray:
    """Footprint corners of annotated boxes, using each box's own dimensions.

    :param boxes: (N, >=7) rows of the openscene annotation layout, where columns
        0, 1 are the centre, 6 is the heading and 3, 4 are length and width. Unlike
        the ego, an annotation's pose is already at the box centre.
    :return: (N, 4, 2)
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return np.zeros((0, 4, 2))

    cx, cy, heading = boxes[:, 0], boxes[:, 1], boxes[:, 6]
    half_length, half_width = boxes[:, 3] / 2.0, boxes[:, 4] / 2.0
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    signs_x = np.array([1.0, -1.0, -1.0, 1.0])
    signs_y = np.array([1.0, 1.0, -1.0, -1.0])
    dx = half_length[:, None] * signs_x
    dy = half_width[:, None] * signs_y

    x = cx[:, None] + cos_h[:, None] * dx - sin_h[:, None] * dy
    y = cy[:, None] + sin_h[:, None] * dx + cos_h[:, None] * dy
    return np.stack([x, y], axis=-1)


class CareClips:
    """Every window of every clip in a CARE export, loadable one at a time.

    Construction reads only the frame dicts (a few MB); images are read per window
    in `features`, so building the index is cheap enough to do in every process.
    """

    def __init__(
        self,
        care_root: Path,
        num_history_frames: int = 4,
        num_future_frames: int = 10,
        stride: int = 1,
    ):
        self.meta_dir = Path(care_root) / "openscene_meta_datas"
        self.sensor_dir = Path(care_root) / "sensor_blobs"
        if not self.meta_dir.is_dir():
            raise FileNotFoundError(
                f"{self.meta_dir} does not exist -- run scripts/vis3d/generate_ras_logs.py"
            )
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.window = num_history_frames + num_future_frames

        self._frames: Dict[Path, List[Dict]] = {}
        self.scenes: List[CareScene] = []
        # sorted(), not iterdir(): the scene order has to be reproducible across
        # processes and across rounds, because the train/val split and the seeded
        # sampling of regular data are both keyed on position in this list.
        for clip_pickle in sorted(self.meta_dir.glob("*.pkl")):
            frames = self._load(clip_pickle)
            last_start = len(frames) - self.window
            for start in range(0, last_start + 1, stride):
                current = frames[start + num_history_frames - 1]
                self.scenes.append(
                    CareScene(
                        token=str(current["token"]),
                        log_name=str(current["log_name"]),
                        clip_pickle=clip_pickle,
                        start_index=start,
                    )
                )

    def _load(self, clip_pickle: Path) -> List[Dict]:
        if clip_pickle not in self._frames:
            with open(clip_pickle, "rb") as handle:
                self._frames[clip_pickle] = pickle.load(handle)
        return self._frames[clip_pickle]

    def __len__(self) -> int:
        return len(self.scenes)

    def tokens(self) -> List[str]:
        return [scene.token for scene in self.scenes]

    # ------------------------------------------------------------------ per window

    def agent_input(self, scene: CareScene, sensor_config):
        """The history the model sees, as the devkit's `AgentInput`.

        Mirrors `AgentInput.from_scene_dict_list` field for field, minus the
        unconditional `Path(lidar_path)` that stops it on these logs. Ego poses are
        relative to the *last* history frame, which is the devkit's convention and
        what the checkpoint was trained with.
        """
        from navsim.common.dataclasses import AgentInput, Cameras, EgoStatus, Lidar

        frames = self._load(scene.clip_pickle)
        history = frames[scene.start_index : scene.start_index + self.num_history_frames]

        global_poses = np.array([_pose_of(frame) for frame in history])
        local_poses = _relative_poses(global_poses, global_poses[-1])

        ego_statuses, cameras, lidars = [], [], []
        for index, frame in enumerate(history):
            dynamic_state = frame["ego_dynamic_state"]
            ego_statuses.append(
                EgoStatus(
                    ego_pose=np.array(local_poses[index], dtype=np.float32),
                    ego_velocity=np.array(dynamic_state[:2], dtype=np.float32),
                    ego_acceleration=np.array(dynamic_state[2:], dtype=np.float32),
                    driving_command=frame["driving_command"],
                )
            )
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=self.sensor_dir,
                    camera_dict=frame["cams"],
                    sensor_names=sensor_config.get_sensors_at_iteration(index),
                )
            )
            # Empty by construction: these logs have no lidar and the RAP sensor
            # config asks for none, so there is nothing to load and nothing that
            # reads it downstream.
            lidars.append(Lidar())

        return AgentInput(ego_statuses, cameras, lidars)

    def features(self, scene: CareScene, feature_builder, sensor_config) -> Dict[str, torch.Tensor]:
        """Model input tensors for one window, built by RAP's own feature builder."""
        return feature_builder.compute_features(self.agent_input(scene, sensor_config))

    def human_trajectory(self, scene: CareScene) -> np.ndarray:
        """The recorded ego future in the current ego frame, (num_future_frames, 3).

        On a CARE clip this is the trajectory that *crashed*. It goes into the
        buffer as a scored candidate -- it is a real trajectory with a real score,
        and often the most informative one in the window -- but never as an
        imitation target; see the trajectory_weight comment in rl/config.py.
        """
        frames = self._load(scene.clip_pickle)
        first = scene.start_index + self.num_history_frames - 1
        poses = np.array(
            [_pose_of(frames[i]) for i in range(first, first + self.num_future_frames + 1)]
        )
        return _relative_poses(poses[1:], poses[0]).astype(np.float32)

    def agent_corners(self, scene: CareScene) -> np.ndarray:
        """Other agents' footprints over the future horizon, in the current ego frame.

        :return: (N, num_future_frames, 4, 2), zero-padded to the busiest timestep.

        Row `n` is NOT one physical agent. The CARE export re-hashes `track_tokens`
        and `instance_tokens` every frame -- measured directly: consecutive frames
        of a clip share zero tokens of either kind -- so the boxes carry no temporal
        association at all, and any layout that claims one would be inventing it.
        Rows are therefore per timestep and independent, padded with zeros where a
        timestep has fewer boxes than the busiest one.

        That costs nothing here: the collision and TTC tests in rl/scoring.py
        intersect each timestep's ego footprint against that timestep's boxes and
        never compare a box to itself at another time. The association would only
        be needed to report *which* agent was hit, which nothing in this pipeline
        does. It does mean a collision label is only as good as the per-frame
        detections behind it -- a box that flickers out for one frame opens a
        one-timestep hole in the obstacle.
        """
        frames = self._load(scene.clip_pickle)
        first = scene.start_index + self.num_history_frames - 1
        origin = _pose_of(frames[first])

        per_step: List[np.ndarray] = []
        for step in range(1, self.num_future_frames + 1):
            frame = frames[first + step]
            boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64)
            if len(boxes) == 0:
                per_step.append(np.zeros((0, 4, 2)))
                continue

            # The boxes are in that frame's own ego frame; move them into the
            # current one. Composing the two SE2s is the whole transform: rotate
            # the box centre by the relative yaw, translate by the relative
            # position, and add the relative yaw to the box heading.
            delta = _relative_poses(_pose_of(frame)[None], origin)[0]
            cos_d, sin_d = np.cos(delta[2]), np.sin(delta[2])
            moved = boxes.copy()
            moved[:, 0] = delta[0] + cos_d * boxes[:, 0] - sin_d * boxes[:, 1]
            moved[:, 1] = delta[1] + sin_d * boxes[:, 0] + cos_d * boxes[:, 1]
            moved[:, 6] = boxes[:, 6] + delta[2]
            per_step.append(agent_box_corners(moved))

        width = max((len(step) for step in per_step), default=0)
        result = np.zeros((width, self.num_future_frames, 4, 2))
        for step_index, corners in enumerate(per_step):
            result[: len(corners), step_index] = corners
        return result
