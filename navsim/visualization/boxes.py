"""Reading the orientation off a 3D box row.

The column layout itself lives in `navsim.common.enums.BoundingBoxIndex`, which
is what the agent feature builders already index through; this module only adds
what drawing a box needs on top of it -- accepting the older seven-column rows,
and turning roll/pitch/yaw into a rotation matrix.

A box is nine numbers::

    [x, y, z,  l, w, h,  roll, pitch, yaw]
     center    size      orientation

Rows used to be seven, ending at a single heading. `as_rpy` widens one of those
by setting roll and pitch to zero, which is what a yaw-only box always meant,
so boxes exported before the change render exactly as they did and the extra
two angles are picked up as soon as the exporter starts writing nine.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from navsim.common.enums import BoundingBoxIndex

CENTER = BoundingBoxIndex.POSITION
DIMS = BoundingBoxIndex.DIMENSION
RPY = BoundingBoxIndex.ORIENTATION

X, Y, Z = BoundingBoxIndex.X, BoundingBoxIndex.Y, BoundingBoxIndex.Z
LENGTH, WIDTH, HEIGHT = BoundingBoxIndex.LENGTH, BoundingBoxIndex.WIDTH, BoundingBoxIndex.HEIGHT
ROLL, PITCH, YAW = BoundingBoxIndex.ROLL, BoundingBoxIndex.PITCH, BoundingBoxIndex.YAW

BOX_DIM = BoundingBoxIndex.size()   # 9
LEGACY_BOX_DIM = 7                  # [x, y, z, l, w, h, yaw]


def as_rpy(boxes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return boxes as (N, 9), widening legacy (N, 7) rows with roll=pitch=0.

    Always returns a copy, so callers may write into the result -- several do,
    to fold the ego pose into the boxes -- without touching the annotations
    they were handed.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, BOX_DIM), dtype=np.float64)
    boxes = boxes.reshape(-1, boxes.shape[-1])

    if boxes.shape[1] == BOX_DIM:
        return boxes.copy()
    if boxes.shape[1] == LEGACY_BOX_DIM:
        widened = np.zeros((len(boxes), BOX_DIM), dtype=np.float64)
        widened[:, CENTER] = boxes[:, 0:3]
        widened[:, DIMS] = boxes[:, 3:6]
        widened[:, YAW] = boxes[:, 6]
        return widened
    raise ValueError(
        f"expected boxes with {LEGACY_BOX_DIM} or {BOX_DIM} columns, got shape {boxes.shape}"
    )


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> npt.NDArray[np.float64]:
    """Rotation for one box, as R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Intrinsic Z-Y-X, the convention `renderer.yaw_to_rot` already implied: with
    roll = pitch = 0 this is exactly Rz(yaw), so widened legacy boxes come out
    bit-identical to what the yaw-only path produced.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def rot_to_rpy(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse of `rpy_to_rot`, as [roll, pitch, yaw].

    Gimbal lock (|pitch| = 90 deg) collapses roll and yaw onto one axis; there
    the split between them is arbitrary and roll is pinned to 0.
    """
    R = np.asarray(R, dtype=np.float64)
    sp = -R[2, 0]
    if abs(sp) >= 1.0 - 1e-9:
        pitch = np.copysign(np.pi / 2.0, sp)
        return np.array([0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])])
    return np.array([
        np.arctan2(R[2, 1], R[2, 2]),
        np.arcsin(sp),
        np.arctan2(R[1, 0], R[0, 0]),
    ])
