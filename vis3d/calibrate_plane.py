#!/usr/bin/env python3
"""Road-plane pitch and camera height from lane geometry alone.

    python calibrate_plane.py --clip ../data/test/changelane        # report
    python calibrate_plane.py --clip ../data/test/changelane --write # store it

WHY NOT calibrate_ground.py

That module reads pitch off the lane vanishing point, which needs a straight
road: a bend moves the VP by far more than any mount is tilted, and its own
docstring says so. On changelane -- a lane-change clip -- it returns -0.96 deg
against a true +0.70, and that 1.7 deg is not a rounding error downstream.
Range on the road plane goes as fy*h/(v - v_horizon), so a pitch error shifts
the horizon row and rescales every range non-linearly. Measured end to end on
changelane, feeding the two pitches through detect_dashes.py + dash_odometry.py:

    pitch +0.70 (this module)      83.8 km/h against a true 87.9
    pitch -0.96 (lane VP)         152.0 km/h against a true 87.9

WHAT THIS MEASURES INSTEAD

Two physical facts, neither of which has a degenerate optimum:

  pitch   lane lines are PARALLEL, so on a correctly calibrated plane the gap
          between two of them must not change with range. Too little pitch and
          the far gap opens, too much and it closes -- the drift passes through
          zero at the true value, so the estimate comes from a SIGN CHANGE
          rather than from maximising something. That distinction matters here:
          picking the pitch that maximised a correlation peak selected -6 deg,
          where the bird's-eye patch degenerates to a constant profile that
          correlates 0.99 with itself and reports a stationary car.
  height  that gap, once the lines are parallel, must equal the real lane width.
          Height enters the back-projection linearly, so this is one division.

LIMITS

Validated on one clip, because changelane is the only one with ground truth to
validate against. It needs lane_masks with two lines visible over a usable range
band, so it is silent on clips with a single line or none. Yaw is taken from
calibrate_ground.py and roll is held at zero -- on a forward dashcam both are
second-order next to pitch.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("calibrate_plane needs opencv") from error

# Range band the lane lines are measured over. This was 10-34 m, which is a
# motorway's geometry: on an urban clip the visible road is mostly nearer than
# 10 m, so the band came back empty and EVERY frame failed at any sane pitch --
# on ambulance, all 30 sampled frames at pitch >= 0. Widened so a junction and a
# motorway both have something in it.
# Outer bounds only. The band actually used is derived per frame from where the
# mask projects (see _band), because a fixed one cannot serve both a motorway
# and a junction: 10-34 m was empty on every urban frame, and widening it to
# 5-45 m globally then broke changelane, which had been the one clip that worked.
Z_LO, Z_HI = 4.0, 50.0
BAND_PERCENTILES = (15, 85)
LANE_WIDTH_M = 3.65

# Windscreen mounts are not all near-level: calibrate_ground's VP put ambulance
# at +9.86 deg, outside the old -8..+4 sweep, so the true plane could not be
# reached however good the measurement was.
PITCH_SWEEP = np.arange(-14.0, 14.01, 0.5)

MIN_FRAMES = 4

MIN_SWEEP_SPAN_DEG = 3.0

# Loosened along with the band: a subpar measurement that the height guard below
# can still sanity-check beats no measurement at all.
MIN_POINTS_PER_HALF = 30
MIN_POINTS_PER_LINE = 8

# The adaptive band has to span enough range for "does the gap change with
# distance" to mean anything.
MIN_BAND_SPAN_M = 8.0

# A windscreen-mounted dashcam sits roughly at the driver's eyeline. Outside this
# the "lane gap" is not a lane -- see the note in estimate().
MIN_CAMERA_HEIGHT_M, MAX_CAMERA_HEIGHT_M = 1.15, 1.75

# Used when nothing can be measured. A windscreen mount sits roughly at the
# driver's eyeline; level is the least-wrong assumption absent evidence.
NOMINAL_PITCH_DEG, NOMINAL_HEIGHT_M = 0.0, 1.35

# Quality ladder, worst to best. Written into road_plane.json as `quality` so a
# guessed plane can never be mistaken for a measured one downstream.
QUALITY_ORDER = ("nominal", "vanishing-point", "vehicles", "approximate", "measured")
PROBE_HEIGHT = 1.3          # any value works; height cancels out of the pitch step


def _project(mask, inverse_k, rotation_t, height):
    ys, xs = np.nonzero(mask > 127)
    if len(ys) < 200:
        return None
    step = max(1, len(ys) // 4000)
    ys, xs = ys[::step], xs[::step]
    rays = inverse_k @ np.c_[xs, ys, np.ones(len(xs))].T
    point = rotation_t @ rays
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = height / point[1]
    ok = np.isfinite(scale) & (scale > 0)
    x, z = (point[0] * scale)[ok], (point[2] * scale)[ok]
    keep = (z > Z_LO) & (z < Z_HI)
    return x[keep], z[keep]


def _band(z):
    """Near/far split taken from the mask's own range spread, not a fixed guess."""
    # A pitch far from the truth can push every projected point outside the outer
    # bounds, leaving nothing to take percentiles of.
    if len(z) < 2 * MIN_POINTS_PER_HALF:
        return None
    lo, hi = np.percentile(z, BAND_PERCENTILES)
    if hi - lo < MIN_BAND_SPAN_M:
        return None
    return lo, hi, 0.5 * (lo + hi)


def _gap(x, z):
    """(change in lane gap from the near half to the far half, mean gap)."""
    band = _band(z)
    if band is None:
        return None
    low, high, middle = band
    inside = (z >= low) & (z <= high)
    x, z = x[inside], z[inside]
    centres = []
    for half in (z < middle, z >= middle):
        if half.sum() < MIN_POINTS_PER_HALF:
            return None
        lateral = np.sort(x[half])
        # a gap wider than a metre separates two painted lines
        groups = np.split(lateral, np.flatnonzero(np.diff(lateral) > 1.0) + 1)
        found = np.array([g.mean() for g in groups if len(g) >= MIN_POINTS_PER_LINE])
        if len(found) < 2:
            return None
        centres.append(np.sort(found))
    near, far = centres
    count = min(len(near), len(far))
    if count < 2:
        return None
    near_gaps, far_gaps = np.diff(near[:count]), np.diff(far[:count])
    usable = (near_gaps > 2.0) & (near_gaps < 6.5) & (far_gaps > 2.0) & (far_gaps < 6.5)
    if not usable.any():
        return None
    return float(np.mean(far_gaps[usable] - near_gaps[usable])), \
           float(np.mean((near_gaps[usable] + far_gaps[usable]) / 2))


def estimate(lane_masks_dir, intrinsics, yaw=0.0, lane_width=LANE_WIDTH_M, frames=60):
    """(pitch_deg, height_m, diagnostics) or (None, None, why)."""
    import calibrate_ground as CG
    names = sorted(p.name for p in Path(lane_masks_dir).iterdir() if p.suffix == ".png")
    if not names:
        return None, None, "no lane masks"
    names = names[::max(1, len(names) // frames)][:frames]
    masks = [cv2.imread(str(Path(lane_masks_dir) / n), cv2.IMREAD_GRAYSCALE) for n in names]
    masks = [m for m in masks if m is not None]
    inverse_k = np.linalg.inv(intrinsics)

    curve = []
    for degrees in PITCH_SWEEP:
        rotation_t = (CG._rot_x(np.radians(degrees)) @ CG._rot_y(yaw) @ CG._rot_z(0.0)).T
        drifts, gaps = [], []
        for mask in masks:
            projected = _project(mask, inverse_k, rotation_t, PROBE_HEIGHT)
            if projected is None:
                continue
            measured = _gap(*projected)
            if measured:
                drifts.append(measured[0])
                gaps.append(measured[1])
        if len(drifts) >= MIN_FRAMES:
            curve.append((float(degrees), float(np.median(drifts)), float(np.median(gaps)), len(drifts)))
    if len(curve) < 3:
        return None, None, "too few frames with two measurable lane lines"

    degrees = np.array([c[0] for c in curve])
    drift = np.array([c[1] for c in curve])
    gaps = np.array([c[2] for c in curve])
    crossings = np.flatnonzero(np.diff(np.sign(drift)) != 0)
    if not len(crossings):
        # No bracketing, so no exact answer -- but a subpar plane that survives
        # the height check is worth more than nothing. Take the flattest drift
        # among the pitches whose implied camera height is physical, and say so.
        implied = PROBE_HEIGHT * lane_width / np.maximum(gaps, 1e-6)
        physical = (implied >= MIN_CAMERA_HEIGHT_M) & (implied <= MAX_CAMERA_HEIGHT_M)
        if not physical.any():
            return None, None, (
                "lane gap drift never changes sign over "
                f"{degrees[0]:+.1f}..{degrees[-1]:+.1f} deg, and no pitch in it implies a "
                f"camera {MIN_CAMERA_HEIGHT_M}-{MAX_CAMERA_HEIGHT_M} m above the road")
        j = np.flatnonzero(physical)[np.argmin(np.abs(drift[physical]))]
        return float(degrees[j]), float(implied[j]), dict(
            quality="approximate: no sign change, flattest physical drift",
            residual_drift=float(drift[j]), gap_at_probe_height=float(gaps[j]),
            crossing_between=(float(degrees[j]), float(degrees[j])),
            frames_used=int(curve[j][3]),
            sweep=[(c[0], round(c[1], 4), round(c[2], 3)) for c in curve])

    if degrees[-1] - degrees[0] < MIN_SWEEP_SPAN_DEG:
        return None, None, (f"only {degrees[-1] - degrees[0]:.1f} deg of the sweep had two "
                            f"measurable lane lines; too little to bracket the plane")
    i = crossings[0]
    pitch = degrees[i] - drift[i] * (degrees[i+1] - degrees[i]) / (drift[i+1] - drift[i])
    gap = float(np.interp(pitch, degrees, [c[2] for c in curve]))
    height = PROBE_HEIGHT * lane_width / gap

    # The sign change itself is well behaved -- the sweeps are clean and monotone
    # on every clip tried. What is NOT reliable is that the gap being measured is
    # one lane: the lane masks are over-extended (see the lane pipeline notes), so
    # the grouping can bracket two lanes, or a lane and a barrier. changelane
    # measures 3.47 m and gives a sane 1.37 m camera; close_slam measures 4.87 m
    # and implies a 0.97 m one, which then ran 32% slow because height scales
    # range linearly. A windscreen-mounted dashcam is not below waist height, so
    # an implausible height is the signal that the gap is not a lane.
    if not (MIN_CAMERA_HEIGHT_M <= height <= MAX_CAMERA_HEIGHT_M):
        return None, None, (f"lane gap measured {gap:.2f} m, which for a {lane_width} m lane implies "
                            f"a camera {height:.2f} m above the road -- outside "
                            f"{MIN_CAMERA_HEIGHT_M}-{MAX_CAMERA_HEIGHT_M} m, so the gap being "
                            "measured is not one lane")
    return float(pitch), float(height), dict(
        gap_at_probe_height=gap, crossing_between=(float(degrees[i]), float(degrees[i+1])),
        frames_used=int(curve[i][3]),
        sweep=[(c[0], round(c[1], 4), round(c[2], 3)) for c in curve])


def main():
    parser = argparse.ArgumentParser(description="Road-plane pitch and height from lane geometry.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--lane_width", type=float, default=LANE_WIDTH_M)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--allow_fallback", action="store_true",
                        help="when lane geometry cannot be measured, still write a plane -- "
                             "fitted from the clip's car detections (vehicle_horizon.py), else "
                             "the lane vanishing point if plausible, else a nominal level camera. "
                             "road_plane.json records which, in `quality`, and everything "
                             "downstream carries that label. Without this the clip is skipped.")
    parser.add_argument("--write", action="store_true",
                        help="store the result in <run>/road_plane.json")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_ground as CG

    clip = Path(args.clip).resolve()
    run = clip / args.run
    kpath = Path(args.intrinsics) if args.intrinsics else run / "camera_intrinsics.json"
    k = json.loads(kpath.read_text())
    intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])

    vp = CG.estimate(run / "lane_masks", intrinsics, (k["height"], k["width"]), height=1.3)
    pitch, height, extra = estimate(run / "lane_masks", intrinsics, yaw=vp.yaw,
                                    lane_width=args.lane_width, frames=args.frames)
    print(f"lane VP says pitch {np.degrees(vp.pitch):+.2f} deg, yaw {np.degrees(vp.yaw):+.2f} "
          f"[{vp.source}]")
    import vehicle_horizon as VH
    vehicles, vehicle_failure = VH.estimate(run, intrinsics, (k["height"], k["width"]))
    if vehicles is not None:
        print(f"car detections say pitch {vehicles['pitch_deg']:+.2f} deg, camera "
              f"{vehicles['height_m']:.2f} m ({vehicles['inliers']} of {vehicles['cars']} cars)")
    else:
        print(f"car-detection plane unavailable: {vehicle_failure}")

    quality = "measured"
    if pitch is None:
        reason = extra
        print(f"parallel-lines calibration FAILED: {reason}")
        if not args.allow_fallback:
            raise SystemExit(1)
        # Something is better than nothing, PROVIDED it cannot pass for a
        # measurement. The car fit is ranked first: it put changelane at +0.68
        # deg against a hand-measured +0.70 and needs no paint at all. The
        # vanishing point is a real observation too, just a poor one -- it was
        # 1.7 deg out on changelane, which came out at 152 km/h against a true
        # 88 -- so it ranks above a flat guess and below the cars.
        vp_deg = float(np.degrees(vp.pitch))
        if vehicles is not None:
            quality, pitch, height = "vehicles", vehicles["pitch_deg"], vehicles["height_m"]
            print(f"falling back to the car-detection plane: pitch {pitch:+.2f} deg, "
                  f"height {height:.2f} m (car height {VH.CAR_HEIGHT_M} m assumed)")
        elif abs(vp_deg) <= abs(PITCH_SWEEP[-1]):
            quality, pitch, height = "vanishing-point", vp_deg, NOMINAL_HEIGHT_M
            print(f"falling back to the lane vanishing point: pitch {pitch:+.2f} deg, "
                  f"height {height} m assumed")
        else:
            quality, pitch, height = "nominal", NOMINAL_PITCH_DEG, NOMINAL_HEIGHT_M
            print(f"vanishing point ({vp_deg:+.2f} deg) is outside the plausible sweep too; "
                  f"falling back to a nominal plane: pitch {pitch:+.2f} deg, height {height} m")
        extra = {"quality": quality, "failure": str(reason)}
    else:
        quality = extra.get("quality", "measured") if isinstance(extra, dict) else "measured"
        if str(quality).startswith("approximate"):
            quality = "approximate"
    if quality in ("measured", "approximate"):
        print(f"parallel lines say pitch {pitch:+.2f} deg "
              f"(drift crosses zero between {extra['crossing_between'][0]:+.1f} and "
              f"{extra['crossing_between'][1]:+.1f}, {extra['frames_used']} frames)")
        print(f"lane width {args.lane_width} m implies camera height {height:.2f} m "
              f"(gap measured {extra['gap_at_probe_height']:.2f} m at a probe height "
              f"of {PROBE_HEIGHT})")
    print(f"PLANE QUALITY: {quality}")
    if args.write:
        dest = run / "road_plane.json"
        source = {"measured": "parallel lane lines + lane width",
                  "approximate": "flattest physical drift, no sign change",
                  "vehicles": "car detections: horizon and camera height, car height assumed",
                  "vanishing-point": "lane vanishing point, height assumed",
                  "nominal": "no measurement available; level camera assumed"}[quality]
        dest.write_text(json.dumps(dict(
            pitch_deg=round(float(pitch), 3), height_m=round(float(height), 3),
            yaw_deg=round(float(np.degrees(vp.yaw)), 3), lane_width_m=args.lane_width,
            quality=quality, source=source,
            failure=extra.get("failure") if isinstance(extra, dict) else None,
            vp_pitch_deg=round(float(np.degrees(vp.pitch)), 3),
            # kept even when the lanes won, as an independent cross-check: a
            # measured plane several degrees off the cars' is worth a look
            vehicle_pitch_deg=None if vehicles is None else round(vehicles["pitch_deg"], 3),
            vehicle_height_m=None if vehicles is None else round(vehicles["height_m"], 3)),
            indent=1))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
