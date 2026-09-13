#!/usr/bin/env python3
"""Roadside uprights as landmarks, for the clips that have no lane dashes.

STATUS (2026-09-12): FAILED VALIDATION -- not wired into run_odometry.sh.
Even on a correct plane (vehicle_horizon.py), turn_blocker has contacts in 7%
of frames and integrates 0 m against a true 146 m; four_way -10 m against 34 m.
Pole bases at intersections sit 30-60 m out near the horizon and are occluded
by parked cars and hedges. Densifying to every static 2-D corner below the
horizon (scratch test, fed to dash_odometry.steps) did not rescue it: corners
cluster on lines parallel to the motion, lock at zero shift, and read 1-15% of
true distance on moving clips with stationary medians up to 55 km/h.

    python detect_roadside.py --clip ../data/test/four_way --pitch_deg 1.2 --height 1.4

WHY

detect_dashes.py needs painted dashes, and 23 of the 49 CARE clips do not have
them in view -- night rural roads, residential streets with no paint, and
solid-only highways. But every one of those clips has poles, tree trunks, sign
posts, fence uprights and building corners, and all of them are bolted to the
ground and stay there while the ego drives past.

WHAT IS ACTUALLY MEASURED

Not the object -- its GROUND CONTACT. A pole's base sits on the road plane, so
back-projecting that one pixel gives a metric (x, z) exactly the way a dash's
does, and no assumption about the object's height is needed. Everything above
the contact point is discarded; it is only used to find the base and to confirm
the thing is an upright rather than a shadow edge or a lane line.

HOW IT DIFFERS FROM DASH ODOMETRY, AND WHY THAT MATTERS

Dash odometry divides out the road plane: the ego's advance and the dash pitch
are both longitudinal lengths in the same coordinates, so their ratio is
calibration-free and only the real dash cycle has to be known. Roadside uprights
have no known spacing, so there is nothing to divide by -- **this estimator
inherits the plane's errors directly**. A camera height 10% out is a speed 10%
out. Use it where dashes are unavailable, and prefer dashes where both exist.

WHAT IT WILL GET WRONG

A kerb line or a wall base running PARALLEL to the road is a long horizontal
edge whose "contact point" slides along it as the ego moves -- the aperture
problem again. Only near-vertical segments are kept, and only their lowest
endpoint, which is what makes a pole usable and a kerb not.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("detect_roadside needs opencv") from error

from detect_dashes import ground_from_pixels, pixels_from_ground

Z_NEAR, Z_FAR = 6.0, 40.0

# An upright must be this far off the clip's own centre line to count as
# roadside. Inside it we would be picking up lane paint and road markings, which
# detect_dashes handles properly and this module would mis-read.
MIN_LATERAL_M = 2.2
MAX_LATERAL_M = 18.0

# Near-vertical in the image: a pole leans a little with perspective, a kerb or a
# wall base does not come close.
MAX_TILT_DEG = 22.0
MIN_SEGMENT_PX = 28

# The contact point must sit below the horizon row by this much, or the
# back-projection is unstable and a few pixels of error become tens of metres.
MIN_BELOW_HORIZON_PX = 25


def _horizon_row(intrinsics, rotation):
    """Image row the road plane recedes to; everything above it is not ground."""
    far = pixels_from_ground(np.array([0.0]), np.array([1.0e5]), intrinsics, rotation, 1.0)
    return float(far[0, 1])


def detect(gray, intrinsics, rotation, height, static_mask=None):
    """Ground contacts of roadside uprights: (x, z, height_px, tilt_deg)."""
    inverse_k = np.linalg.inv(intrinsics)
    rotation_t = rotation.T
    image_h, image_w = gray.shape
    horizon = _horizon_row(intrinsics, rotation)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180)
    if static_mask is not None:
        edges = edges * static_mask.astype(np.uint8)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                               minLineLength=MIN_SEGMENT_PX, maxLineGap=6)
    if segments is None or len(segments) == 0:
        return []
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    segments = np.asarray(segments).reshape(-1, 4)

    out = []
    for x1, y1, x2, y2 in segments:
        if y1 == y2:
            continue
        tilt = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
        tilt = min(tilt, 180.0 - tilt)
        if tilt > MAX_TILT_DEG:
            continue
        # the lower endpoint is where the upright meets the ground
        (bx, by) = (x1, y1) if y1 > y2 else (x2, y2)
        if by < horizon + MIN_BELOW_HORIZON_PX or by >= image_h - 2:
            continue
        if static_mask is not None and not static_mask[int(by), int(np.clip(bx, 0, image_w - 1))]:
            continue
        x, z, ok = ground_from_pixels(np.array([[float(bx), float(by)]]),
                                      inverse_k, rotation_t, height)
        if not ok[0]:
            continue
        x, z = float(x[0]), float(z[0])
        if not (Z_NEAR <= z <= Z_FAR):
            continue
        if not (MIN_LATERAL_M <= abs(x) <= MAX_LATERAL_M):
            continue
        out.append((x, z, float(abs(y2 - y1)), float(tilt)))

    # One upright yields several collinear segments; merge anything sharing a
    # ground position, keeping the tallest, or a single pole counts as five.
    out.sort(key=lambda r: -r[2])
    kept = []
    for candidate in out:
        if all(abs(candidate[0] - k[0]) > 0.6 or abs(candidate[1] - k[1]) > 1.2 for k in kept):
            kept.append(candidate)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--pitch_deg", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_ground as CG
    from estimate_ego_motion import _static_mask, _load_object_masks

    clip = Path(args.clip).resolve()
    run = clip / args.run
    k = json.loads((Path(args.intrinsics) if args.intrinsics
                    else run / "camera_intrinsics.json").read_text())
    intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    shape = (k["height"], k["width"])

    plane = run / "road_plane.json"
    pitch, height, quality, yaw_deg = args.pitch_deg, args.height, "given", 0.0
    if plane.exists():
        stored = json.loads(plane.read_text())
        yaw_deg = float(stored.get("yaw_deg") or 0.0)
        if pitch is None or height is None:
            pitch = stored["pitch_deg"] if pitch is None else pitch
            height = stored["height_m"] if height is None else height
            quality = stored.get("quality", "legacy")
    if pitch is None or height is None:
        raise SystemExit("need --pitch_deg/--height or a road_plane.json")
    # Yaw from road_plane.json rather than re-deriving it from the lane VP: that
    # was the slowest part of this script and is exactly the measurement a
    # paint-less clip cannot make.
    rotation = CG._rot_x(np.radians(pitch)) @ CG._rot_y(np.radians(yaw_deg)) @ CG._rot_z(0.0)
    print(f"pitch {pitch:+.2f} deg, height {height} m, plane quality '{quality}' "
          f"-- this estimator inherits the plane's error directly")

    frames = sorted(p for p in (clip / "frames").iterdir()
                    if p.suffix.lower() in {".jpg", ".png"})[args.start:args.end]
    masks = _load_object_masks(run)
    result = {}
    for path in frames:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        static = _static_mask(masks, path.name, shape) if masks else None
        result[path.stem] = detect(gray, intrinsics, rotation, height, static)

    counts = np.array([len(v) for v in result.values()])
    print(f"{len(result)} frames, {counts.sum()} roadside contacts, "
          f"median {np.median(counts) if len(counts) else 0:.0f}/frame, "
          f"{(counts > 0).mean()*100 if len(counts) else 0:.0f}% of frames have one")
    destination = Path(args.out) if args.out else run / "roadside.json"
    destination.write_text(json.dumps(
        {"pitch_deg": float(pitch), "height_m": float(height), "plane_quality": quality,
         "scale_source": "road plane (NOT divided out -- unlike dash odometry)",
         "frames": {s: [[x, z - 2.0, z + 2.0, z, t] for x, z, _, t in v]
                    for s, v in result.items()}}))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
