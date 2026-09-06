"""The on-disk shape of a 3D box, and the conversions either side of it.

A box in boxes_3d.json is an object with named fields::

    {"x": .., "y": .., "z": ..,                 centre, ego frame, metres
     "length": .., "width": .., "height": ..,   extent, metres
     "roll": .., "pitch": .., "yaw": ..}        orientation, radians

Two older shapes are still read, because clips rendered before this change are
not re-rendered just to be loaded: a bare 7-list ``[x, y, z, l, w, h, yaw]``
and a bare 9-list with roll and pitch appended. Both come back from `from_any`
as the same nine-element array, so a reader never has to know which it got.

The fitting code works in arrays, not dicts -- `_box_corners_ego` and the
heading search index positionally and are hot loops -- so the conversion lives
here at the file boundary rather than in the maths. `to_dict` on the way out,
`from_any` on the way in, arrays in between.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Order matters: it is the positional order of the array form, and the order
# the keys are written in, so a file stays diff-friendly across runs.
KEYS = ("x", "y", "z", "length", "width", "height", "roll", "pitch", "yaw")

BOX_DIM = len(KEYS)          # 9
LEGACY_BOX_DIM = 7           # [x, y, z, l, w, h, yaw]

X, Y, Z = 0, 1, 2
LENGTH, WIDTH, HEIGHT = 3, 4, 5
ROLL, PITCH, YAW = 6, 7, 8

CENTER = slice(0, 3)
DIMS = slice(3, 6)
RPY = slice(6, 9)


def from_any(box) -> np.ndarray:
    """One box -> (9,) float array, from a dict or either legacy list form."""
    if isinstance(box, dict):
        missing = [k for k in KEYS if k not in box]
        if missing:
            raise ValueError(f"box is missing {missing}; got keys {sorted(box)}")
        return np.array([float(box[k]) for k in KEYS], dtype=np.float64)

    values = np.asarray(box, dtype=np.float64).ravel()
    if values.size == BOX_DIM:
        return values
    if values.size == LEGACY_BOX_DIM:
        # Legacy row: yaw sat where roll now does, and there was no roll/pitch.
        widened = np.zeros(BOX_DIM, dtype=np.float64)
        widened[:LEGACY_BOX_DIM - 1] = values[:LEGACY_BOX_DIM - 1]
        widened[YAW] = values[LEGACY_BOX_DIM - 1]
        return widened
    raise ValueError(
        f"box must be a dict or have {LEGACY_BOX_DIM}/{BOX_DIM} values, got {values.size}")


def to_array(boxes) -> np.ndarray:
    """A frame's boxes -> (N, 9) float array. Empty input gives (0, 9)."""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, BOX_DIM), dtype=np.float64)
    return np.stack([from_any(b) for b in boxes])


def to_dict(box) -> dict:
    """One box -> the named-field dict that gets serialised."""
    values = from_any(box)
    return {k: float(v) for k, v in zip(KEYS, values)}


def to_dicts(boxes) -> list:
    """A frame's boxes -> list of named-field dicts."""
    return [to_dict(b) for b in boxes]


def dump(obj, path: Path | str) -> None:
    """Write boxes_3d.json.

    indent=2 throughout: this file is read by hand often enough that the space
    is worth it, and every writer goes through here so a later stage cannot
    silently flatten what an earlier one indented.
    """
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
