#!/usr/bin/env python3
"""Road plane from the other cars in the clip -- no lane paint needed.

    python vehicle_horizon.py --clip ../data/CARE_YTB/turn_blocker

WHY

Every plane source before this one reads lane markings: calibrate_plane.py
needs two parallel lines, calibrate_ground.py's vanishing point needs a pencil
of them. The 23 CARE clips with no dashes are mostly the clips with no usable
paint, so they got the nominal level camera -- and on those clips a level camera
is badly wrong: turn_blocker's road vanishes at row ~770 of 1080, not 540, and
four_way's stored -6.5 deg (the degenerate optimum) puts the horizon below
every pole base. A roadside contact back-projected through either plane is
noise.

But every clip has mask_results_preds.json, and a car is an object of roughly
known height standing on the road. For a car whose bottom edge is at row v_b,
its pixel height is

    h_px = (H_car / h_cam) * (v_b - v_horizon)

-- a straight line in (v_b, h_px) across all the clip's cars. One robust line fit
gives the horizon row (the intercept) and the camera height (H_car / slope).
Nothing about the cars' distance, speed or position is needed.

VALIDATION

    changelane   pitch +0.68 deg against a hand-measured +0.70; height 1.49 m
                 against 1.33 (the car-height prior, 1.55 m, reads ~12% high)
    turn_blocker horizon row 772 against ~770 read off the frame by eye

WHAT IT WILL GET WRONG

The HEIGHT carries the car-height prior linearly: a clip of SUVs and pickups
reads high, a clip of hatchbacks low. The PITCH does not depend on the prior at
all -- the intercept is independent of the slope's scale -- which is why it is
the half to trust. Occluded cars are short for their row and fall off the line
as outliers; truncated ones are excluded before the fit. On a hill the cars
ahead are not on the ego's plane and the horizon shifts with them.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Mixed passenger fleet. Sedans are ~1.45 m, crossovers and SUVs 1.65-1.8 m.
CAR_HEIGHT_M = 1.55
CATEGORIES = ("car",)          # trucks and buses have no useful height prior
MIN_SCORE = 0.5
MIN_HEIGHT_PX = 25
BONNET_FRACTION = 0.93         # a box reaching this far down is the ego's bonnet or cut off
EDGE_PX = 4

MIN_INLIERS = 30
RANSAC_ITERATIONS = 3000
INLIER_REL, INLIER_ABS_PX = 0.12, 3.0

# Outside these the fit is describing something other than a windscreen mount.
MIN_CAMERA_HEIGHT_M, MAX_CAMERA_HEIGHT_M = 0.9, 2.2
MAX_PITCH_DEG = 15.0


def boxes(masks_json, image_hw):
    """(bottom row, pixel height) of every untruncated car detection."""
    from pycocotools import mask as mask_utils

    detections = [d for d in masks_json
                  if d.get("category") in CATEGORIES and d.get("score", 1.0) >= MIN_SCORE]
    if not detections:
        return np.zeros(0), np.zeros(0)
    rle = [dict(size=[int(s) for s in d["mask"]["size"]], counts=d["mask"]["counts"])
           for d in detections]
    x, y, w, h = np.asarray(mask_utils.toBbox(rle)).T
    image_h, image_w = image_hw
    bottom = y + h
    whole = ((x > EDGE_PX) & (x + w < image_w - EDGE_PX) & (y > EDGE_PX)
             & (bottom < BONNET_FRACTION * image_h) & (h > MIN_HEIGHT_PX)
             # a car seen from any angle is at least about as wide as it is tall;
             # a narrower box is a partial occlusion
             & (w > 0.9 * h))
    return bottom[whole], h[whole]


def fit(bottom, height_px, seed=0):
    """(horizon row, slope, inlier count) of h_px = slope * (v_b - horizon), or None."""
    if len(bottom) < MIN_INLIERS:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(RANSAC_ITERATIONS):
        i, j = rng.choice(len(bottom), 2, replace=False)
        if abs(bottom[i] - bottom[j]) < 20:
            continue
        slope = (height_px[i] - height_px[j]) / (bottom[i] - bottom[j])
        if slope <= 0:
            continue
        horizon = bottom[i] - height_px[i] / slope
        inliers = (np.abs(height_px - slope * (bottom - horizon))
                   < INLIER_REL * height_px + INLIER_ABS_PX)
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None or best_count < MIN_INLIERS:
        return None
    design = np.c_[bottom[best], np.ones(best_count)]
    slope, intercept = np.linalg.lstsq(design, height_px[best], rcond=None)[0]
    if slope <= 0:
        return None
    return float(-intercept / slope), float(slope), best_count


def estimate(run_dir, intrinsics, image_hw, car_height=CAR_HEIGHT_M):
    """{"pitch_deg", "height_m", "horizon_row", "cars", "inliers"} or (None, reason)."""
    path = Path(run_dir) / "mask_results_preds.json"
    if not path.exists():
        return None, f"no {path.name}"
    bottom, height_px = boxes(json.loads(path.read_text()), image_hw)
    result = fit(bottom, height_px)
    if result is None:
        return None, f"only {len(bottom)} whole car detections, too few for a fit"
    horizon, slope, inliers = result
    fy, cy = intrinsics[1, 1], intrinsics[1, 2]
    # calibrate_ground's convention: the horizon sits at cy + fy * tan(-pitch)
    pitch = -float(np.degrees(np.arctan((horizon - cy) / fy)))
    height = car_height / slope
    if not (MIN_CAMERA_HEIGHT_M <= height <= MAX_CAMERA_HEIGHT_M) or abs(pitch) > MAX_PITCH_DEG:
        return None, (f"implausible fit: pitch {pitch:+.1f} deg, camera {height:.2f} m "
                      f"from {inliers} of {len(bottom)} cars")
    return {"pitch_deg": pitch, "height_m": float(height), "horizon_row": horizon,
            "cars": int(len(bottom)), "inliers": inliers}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--car_height", type=float, default=CAR_HEIGHT_M)
    args = parser.parse_args()
    run = Path(args.clip).resolve() / args.run
    k = json.loads((Path(args.intrinsics) if args.intrinsics
                    else run / "camera_intrinsics.json").read_text())
    intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    plane, reason = estimate(run, intrinsics, (k["height"], k["width"]), args.car_height)
    if plane is None:
        raise SystemExit(f"vehicle horizon FAILED: {reason}")
    print(f"horizon row {plane['horizon_row']:.0f}, pitch {plane['pitch_deg']:+.2f} deg, "
          f"camera {plane['height_m']:.2f} m  ({plane['inliers']} of {plane['cars']} cars)")


if __name__ == "__main__":
    main()
