#!/usr/bin/env python3
"""Ego speed from the road surface itself -- for clips with no usable lane dashes.

    python road_odometry.py --clip ../data/CARE_YTB/yield_runway

WHY

dash_odometry.py needs dashed lane paint, and 23 of the 49 CARE clips do not
give it one. Every road still carries stationary structure that crosses the
direction of travel: turn arrows, crosswalk bars, stop bars, dash ends, joints,
patches. This module warps consecutive frames onto the road plane (a bird's-eye
view at 0.1 m per pixel) and finds the forward shift that carries one onto the
other: the ego's speed is how fast that stationary road moves toward it.

HOW

1. The gradient ALONG the road only. Lane lines, kerbs and tyre polish run
   parallel to the motion and carry no forward information; their derivative
   along the road is ~0, so they drop out. The road direction is measured per
   frame from the warp's structure tensor, because the plane carries no yaw.
2. Painted markings weighted up. A pixel brighter than the road around it at
   marking scale (white top-hat) counts 1 + PAINT_GAIN times an asphalt pixel:
   markings are high-contrast and not self-similar, asphalt texture is both.
3. Dark non-paint pixels left out altogether (black-hat). On four_way, the
   shadows of cars crossing in front while the ego waited moved steadily enough
   to pass the consistency check and integrated to -16 m of phantom reverse.
4. The whole road corridor -- leftmost to rightmost drivable pixel per row --
   not only the ego's lane, so a turn arrow in the next lane counts.
5. The band starts at the bonnet, not at 9 m: the drivable mask's bottom row
   when it stops short of the frame bottom, else the top of the camera-fixed
   (frozen-pixel) band, else 9 m. A stop bar or crosswalk under a waiting car is
   5-9 m out. The fallback matters: where the drivable mask runs to the last row
   the "bonnet" came out at the frame bottom and the band started on the ego's
   own hood (too_close 43% -> 29%, turn_blocker 101% -> 73% before the guard).
6. A whole-patch NCC over a symmetric search (+-4 m forward, +-1.5 m sideways
   per frame of baseline), and 1/2/3-frame consistency: a step is kept only
   when at least three estimates agree within AGREE_M. Nothing depends on the
   sign of the step, so noise cannot be rectified into motion.

The plane is the car-detection plane (vehicle_horizon.py) first, road_plane.json
only when there are too few cars.

VALIDATION (2026-09-12, 11 clips with burned-in OSD speed; distance filled over
unmatched frames and scored up to the last OSD sample; "speed" is the median on
measured moving frames against truth on the same frames)

    clip            distance   speed    note
    turn_blocker      101%      82%
    turn_overtake     106%      97%
    yield_runway      109%     103%     only 18% of moving frames measured
    ambulance         119%     137%
    close_bike        127%      89%     only 21% of moving frames measured
    reserved_lane     133%      89%
    four_way          150%     165%     creeping junction; stops read 0.0
    exit_now           54%      77%     only 12% of moving frames measured
    too_close          44%       7%     view blocked by traffic; edge lines only
    close_slam         34%       1%     wet road: reflections lock at zero shift
    changelane         31%      10%     highway; only 8% of moving frames measured

Tried on the same clips and not kept: a stop gate (zero shift within 90% of the
peak -> step 0) fired on moving frames of close_slam and changelane and cut
real motion; masking edges fixed in the view removed <1% of pixels anywhere; a
longer view (to 50 m) fixed changelane (100%) and too_close (57%) but
overshot the slow clips (four_way 170%, reserved_lane 173%) and broke the
sparse ones (close_bike 27%, exit_now 19%) -- the view length wants to depend
on speed.

WHAT IT WILL GET WRONG

- Highway speed: at ~2.4 m per frame the 2- and 3-frame matches lose overlap
  and the consistency check drops most moving frames.
- Wet roads: reflections of the scene barely move and win the match at zero.
- It inherits the plane's height directly -- a 10% height error is a 10% speed
  error. Unlike dash odometry, there is no known length to divide it out.
- Per frame it is noisy; read the median and the distance, and smooth before
  using a per-frame speed. Where few moving frames are measured, the distance
  is mostly interpolation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("road_odometry needs opencv") from error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_dashes import ground_from_pixels, pixels_from_ground  # noqa: E402

RES_M = 0.1
X_HALF_M, Z_FAR_M = 10.0, 40.0
# The template runs from 1 m past the band's near edge to TEMPLATE_FAR_M, so the
# search can move it +-SEARCH inside the band.
TEMPLATE_NEAR_MARGIN_M, TEMPLATE_FAR_M, TEMPLATE_HALF_M = 1.0, 32.0, 7.0
SEARCH_Z_M, SEARCH_X_M = 4.0, 1.5            # per frame of baseline; 4 m is 144 km/h at 10 Hz
AMBIGUITY_RATIO = 0.85                        # a rival peak >0.6 m away this strong: reject
MIN_PEAK = 0.05
MIN_TEMPLATE_COVERAGE = 0.15
MAX_ROAD_ANGLE_DEG = 25.0

# Every per-frame distance above (search, agreement) was tuned on 10 Hz frames.
# At another rate they are scaled by REFERENCE_HZ / hz, so they stay the same
# distance per unit TIME -- but nothing below 10 Hz has been validated.
REFERENCE_HZ = 10.0

BASELINES = (1, 2, 3)
AGREE_M = 0.5
MIN_AGREEING = 3

VEHICLE_DILATE_PX = 31
ROAD_DILATE_PX = 21
FROZEN_STD = 6.0                              # grey-level sd over the clip below which a pixel is camera-fixed
GRADIENT_CLIP_SIGMA = 4.0                     # one very bright mark must not dominate the NCC

# Painted markings and dark non-paint.
PAINT_GAIN = 4.0                              # a paint pixel counts 1 + PAINT_GAIN times an asphalt pixel
TOPHAT_M = 1.2                                # markings narrower than this stand out of the local road
PAINT_SIGMA = 4.0                             # robust sigmas above the band's own median
PAINT_MIN_GREY = 12.0

# Near edge of the band.
NEAR_DEFAULT_M = 9.0                          # used when no bonnet edge can be found
NEAR_MIN_M = 4.0
BONNET_MARGIN_M = 0.5
BONNET_EDGE_CLEARANCE_PX = 6                  # a drivable mask ending closer than this to the bottom has no bonnet in it
FROZEN_BONNET_MAX_FRACTION = 0.95             # a frozen band starting lower than this is OSD text, not a bonnet


def load_plane(run, intrinsics, image_hw, source="auto"):
    """(pitch_deg, height_m, description) of the road plane to warp through."""
    import vehicle_horizon as VH

    stored = run / "road_plane.json"
    if source in ("auto", "vehicles"):
        plane, reason = VH.estimate(run, intrinsics, image_hw)
        if plane is not None:
            return plane["pitch_deg"], plane["height_m"], (
                f"vehicles ({plane['inliers']} of {plane['cars']} cars)")
        if source == "vehicles":
            raise SystemExit(f"vehicle plane unavailable: {reason}")
        print(f"vehicle plane unavailable ({reason}); using road_plane.json")
    if stored.exists():
        data = json.loads(stored.read_text())
        return data["pitch_deg"], data["height_m"], (
            f"road_plane.json ({data.get('quality', 'legacy')})")
    raise SystemExit("no road plane: no car detections to fit and no road_plane.json")


def _frame_names(frames_dir):
    return sorted(p.name for p in Path(frames_dir).iterdir()
                  if p.suffix.lower() in {".jpg", ".png"})


def near_limit(clip, run, intrinsics, image_hw, pitch_deg, height_m):
    """(near edge of the band in metres, where it came from)."""
    from calibrate_ground import _rot_x

    image_h, image_w = image_hw
    rows = []
    masks = sorted((run / "drivable_masks").glob("*.png"))
    for path in masks[::max(1, len(masks) // 40)]:
        mask = cv2.imread(str(path), 0)
        if mask is None:
            continue
        has = np.flatnonzero((mask[:, image_w // 3: 2 * image_w // 3] > 0).any(1))
        if len(has):
            rows.append(has[-1])
    drivable_row = float(np.median(rows)) if rows else float(image_h)

    bonnet_row, source = None, None
    if drivable_row < image_h - BONNET_EDGE_CLEARANCE_PX:
        bonnet_row, source = drivable_row, "drivable mask"
    else:
        frames = clip / "frames"
        names = _frame_names(frames)
        sample = np.stack([cv2.GaussianBlur(cv2.imread(str(frames / n), 0), (9, 9), 0)
                           .astype(np.float32) for n in names[::max(1, len(names) // 40)]])
        frozen_centre = (sample.std(0) < FROZEN_STD)[:, image_w // 3: 2 * image_w // 3].mean(1) > 0.5
        row = image_h - 1
        while row > image_h // 2 and frozen_centre[row]:
            row -= 1
        if row < image_h * FROZEN_BONNET_MAX_FRACTION:
            bonnet_row, source = float(row), "frozen pixels"
    if bonnet_row is None:
        return NEAR_DEFAULT_M, f"no bonnet edge found: default {NEAR_DEFAULT_M:g} m"

    _, z, ok = ground_from_pixels(np.array([[intrinsics[0, 2], bonnet_row]]),
                                  np.linalg.inv(intrinsics), _rot_x(np.radians(pitch_deg)).T,
                                  height_m)
    z = float(z[0]) if ok[0] else NEAR_DEFAULT_M
    return float(np.clip(z + BONNET_MARGIN_M, NEAR_MIN_M, NEAR_DEFAULT_M)), source


def road_corridor(road, image_w):
    """Everything between the leftmost and rightmost drivable pixel of each row,
    carried above the top road row -- so markings in adjacent lanes count."""
    has = road.any(1)
    left = np.where(has, road.argmax(1), image_w)
    right = np.where(has, image_w - 1 - road[:, ::-1].argmax(1), -1)
    top = np.flatnonzero(has)
    if len(top):
        left[:top[0]], right[:top[0]] = left[top[0]], right[top[0]]
    columns = np.arange(image_w)[None, :]
    return (columns >= left[:, None]) & (columns <= right[:, None])


class RoadView:
    """Bird's-eye warps of one clip's frames, masked to the road surface."""

    def __init__(self, clip, run, intrinsics, image_hw, pitch_deg, height_m,
                 near_m=NEAR_DEFAULT_M, hz=REFERENCE_HZ):
        from calibrate_ground import _rot_x

        self.frames = clip / "frames"
        self.run = run
        self.image_h, self.image_w = image_hw
        self.names = _frame_names(self.frames)
        self.near_m = near_m
        self.frame_scale = REFERENCE_HZ / float(hz)

        xs = np.arange(-X_HALF_M, X_HALF_M + 1e-9, RES_M)
        zs = np.arange(Z_FAR_M, near_m - 1e-9, -RES_M)
        grid_x, grid_z = np.meshgrid(xs, zs)
        uv = pixels_from_ground(grid_x.ravel(), grid_z.ravel(), intrinsics,
                                _rot_x(np.radians(pitch_deg)), height_m)
        self.uv = uv.reshape(len(zs), len(xs), 2).astype(np.float32)
        self.inside = ((self.uv[..., 0] >= 0) & (self.uv[..., 0] < self.image_w - 1)
                       & (self.uv[..., 1] >= 0) & (self.uv[..., 1] < self.image_h - 1))

        self.row0 = int((Z_FAR_M - TEMPLATE_FAR_M) / RES_M)
        self.row1 = int((Z_FAR_M - (near_m + TEMPLATE_NEAR_MARGIN_M)) / RES_M)
        self.col0 = int((X_HALF_M - TEMPLATE_HALF_M) / RES_M)
        self.col1 = int((X_HALF_M + TEMPLATE_HALF_M) / RES_M)

        # Pixels that barely change over the whole clip: bonnet, OSD text, mounts.
        stride = max(1, len(self.names) // 40)
        sample = np.stack([cv2.GaussianBlur(cv2.imread(str(self.frames / n), 0), (9, 9), 0)
                           .astype(np.float32) for n in self.names[::stride]])
        self.frozen = cv2.dilate((sample.std(0) < FROZEN_STD).astype(np.uint8),
                                 np.ones((25, 25), np.uint8))

        self.detections = {}
        masks = run / "mask_results_preds.json"
        if masks.exists():
            for entry in json.loads(masks.read_text()):
                self.detections.setdefault(entry["frame"], []).append(entry["mask"])
        self.paint_share = []
        self._cache = {}

    def warp(self, image, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(image, self.uv[..., 0], self.uv[..., 1], interpolation, borderValue=0)

    def road_angle(self, index):
        """Direction of the road in the warp, radians from straight ahead."""
        image = cv2.imread(str(self.frames / self.names[index]), 0)
        warped = cv2.GaussianBlur(self.warp(image).astype(np.float32), (0, 0), 1.0)
        gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
        usable = self.inside & (self.warp(self.frozen, cv2.INTER_NEAREST) == 0)
        magnitude = np.hypot(gx, gy)
        strong = usable & (magnitude > np.percentile(magnitude[usable], 90))
        jxx, jyy = (gx[strong] ** 2).sum(), (gy[strong] ** 2).sum()
        jxy = (gx[strong] * gy[strong]).sum()
        angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        limit = np.radians(MAX_ROAD_ANGLE_DEG)
        return float(np.clip(angle, -limit, limit))

    def gradient(self, index, angle):
        """Marking-weighted along-road gradient of one frame's warp, and its valid mask."""
        key = (index, round(angle, 3))
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) > 12:
            self._cache.clear()
        name = self.names[index]
        image = cv2.imread(str(self.frames / name), 0)
        bad = self.frozen.copy()
        if self.detections.get(name):
            from pycocotools import mask as mask_utils
            for rle in self.detections[name]:
                rle = dict(size=[int(s) for s in rle["size"]], counts=rle["counts"])
                bad |= cv2.dilate(mask_utils.decode(rle),
                                  np.ones((VEHICLE_DILATE_PX, VEHICLE_DILATE_PX), np.uint8))
        valid = self.inside & (self.warp(bad, cv2.INTER_NEAREST) == 0)
        road = cv2.imread(str(self.run / "drivable_masks" / f"{Path(name).stem}.png"), 0)
        if road is not None:
            road = cv2.dilate((road > 0).astype(np.uint8),
                              np.ones((ROAD_DILATE_PX, ROAD_DILATE_PX), np.uint8)) > 0
            corridor = road_corridor(road, self.image_w).astype(np.uint8)
            valid &= self.warp(corridor, cv2.INTER_NEAREST) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        warped = cv2.GaussianBlur(self.warp(image).astype(np.float32), (0, 0), 1.0)
        size = max(3, int(round(TOPHAT_M / RES_M)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

        def above_band(response):
            values = response[valid]
            centre = np.median(values)
            spread = 1.4826 * np.median(np.abs(values - centre)) + 1e-6
            return response > max(centre + PAINT_SIGMA * spread, PAINT_MIN_GREY)

        if valid.sum() < 500:
            self._cache[key] = (None, None)
            return self._cache[key]
        paint = valid & above_band(cv2.morphologyEx(warped, cv2.MORPH_TOPHAT, kernel))
        paint = cv2.dilate(paint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        dark = valid & above_band(cv2.morphologyEx(warped, cv2.MORPH_BLACKHAT, kernel)) & ~paint
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        valid = valid & ~dark
        if valid.sum() < 500:
            self._cache[key] = (None, None)
            return self._cache[key]
        self.paint_share.append(float(paint.sum() / max(valid.sum(), 1)))

        gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
        along = -np.sin(angle) * gx + np.cos(angle) * gy
        values = along[valid]
        along = np.clip((along - values.mean()) / (values.std() + 1e-6),
                        -GRADIENT_CLIP_SIGMA, GRADIENT_CLIP_SIGMA)
        along *= 1.0 + PAINT_GAIN * paint
        along[~valid] = 0
        self._cache[key] = (along, valid)
        return self._cache[key]

    def shift(self, before, valid_before, after, baseline):
        """Forward advance in metres from `before` to `after`, or None if ambiguous."""
        template = before[self.row0:self.row1, self.col0:self.col1]
        mask = valid_before[self.row0:self.row1, self.col0:self.col1].astype(np.float32)
        if mask.mean() < MIN_TEMPLATE_COVERAGE:
            return None
        search_rows = min(int(SEARCH_Z_M * baseline * self.frame_scale / RES_M), after.shape[0])
        search_cols = min(int(SEARCH_X_M * baseline * self.frame_scale / RES_M), after.shape[1])
        padded = np.zeros((after.shape[0] + 2 * search_rows, after.shape[1] + 2 * search_cols),
                          np.float32)
        padded[search_rows:search_rows + after.shape[0],
               search_cols:search_cols + after.shape[1]] = after
        region = padded[self.row0:self.row1 + 2 * search_rows,
                        self.col0:self.col1 + 2 * search_cols]
        score = cv2.matchTemplate(region, template, cv2.TM_CCORR_NORMED, mask=mask)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        row, col = np.unravel_index(np.argmax(score), score.shape)
        peak = score[row, col]
        far = np.abs(np.arange(score.shape[0]) - row) > int(0.6 / RES_M)
        rival = score[far].max() if far.any() else 0.0
        if peak <= MIN_PEAK or rival > AMBIGUITY_RATIO * peak:
            return None
        offset = 0.0
        if 0 < row < score.shape[0] - 1:
            y0, y1, y2 = score[row - 1, col], score[row, col], score[row + 1, col]
            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-9:
                offset = float(np.clip(0.5 * (y0 - y2) / denominator, -1, 1))
        # A road point moves to a nearer (larger) row as the ego advances.
        return (row - search_rows + offset) * RES_M


def estimate(view, end=None):
    """Advance in metres per frame (index i = frame i-1 -> i; index 0 is nan)."""
    count = len(view.names) if end is None else min(end, len(view.names))
    advance = {k: np.full(count, np.nan) for k in BASELINES}     # advance[k][i]: frame i -> i+k
    for i in range(count - 1):
        angle = view.road_angle(i)
        before, valid = view.gradient(i, angle)
        if before is None:
            continue
        for k in BASELINES:
            if i + k >= count:
                continue
            after, _ = view.gradient(i + k, angle)
            if after is None:
                continue
            step = view.shift(before, valid, after, k)
            if step is not None:
                advance[k][i] = step

    one, two, three = advance[1], advance.get(2), advance.get(3)
    steps = np.full(count, np.nan)
    for i in range(count - 1):
        votes = []
        if np.isfinite(one[i]):
            votes.append(one[i])
        if i + 1 < count and np.isfinite(two[i]) and np.isfinite(one[i + 1]):
            votes.append(two[i] - one[i + 1])
        if i >= 1 and np.isfinite(two[i - 1]) and np.isfinite(one[i - 1]):
            votes.append(two[i - 1] - one[i - 1])
        if np.isfinite(two[i]):
            votes.append(two[i] / 2)
        for back in range(3):
            if i - back >= 0 and np.isfinite(three[i - back]):
                votes.append(three[i - back] / 3)
        votes = np.array(votes)
        if len(votes) < MIN_AGREEING:
            continue
        agreeing = votes[np.abs(votes - np.median(votes)) <= AGREE_M * view.frame_scale]
        if len(agreeing) >= MIN_AGREEING:
            steps[i + 1] = float(np.median(agreeing))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--plane", choices=("auto", "vehicles", "file"), default="auto",
                        help="auto: car-detection plane, else road_plane.json")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out", default=None, help="default: <run>/road_speed.json")
    args = parser.parse_args()

    clip = Path(args.clip).resolve()
    run = clip / args.run
    k = json.loads((Path(args.intrinsics) if args.intrinsics
                    else run / "camera_intrinsics.json").read_text())
    intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    image_hw = (k["height"], k["width"])

    pitch, height, plane_source = load_plane(run, intrinsics, image_hw, args.plane)
    near_m, near_source = near_limit(clip, run, intrinsics, image_hw, pitch, height)
    print(f"plane: pitch {pitch:+.2f} deg, height {height:.2f} m [{plane_source}] -- "
          f"the speed scales with this height directly")
    print(f"band starts at {near_m:.1f} m [{near_source}]")
    if abs(args.hz - REFERENCE_HZ) > 0.5:
        print(f"WARNING: frames are {args.hz:g} Hz; search and agreement are scaled from the "
              f"{REFERENCE_HZ:g} Hz they were tuned at, but nothing below {REFERENCE_HZ:g} Hz "
              f"has been validated against ground truth")
    view = RoadView(clip, run, intrinsics, image_hw, pitch, height, near_m, args.hz)
    steps = estimate(view, args.end)

    kmh = steps * args.hz * 3.6
    matched = np.isfinite(kmh)
    moving = matched & (kmh > 10)
    filled = np.interp(np.arange(len(kmh)), np.flatnonzero(matched), kmh[matched]) \
        if matched.sum() > 1 else np.zeros(len(kmh))
    paint = float(np.median(view.paint_share)) if view.paint_share else 0.0
    print(f"{matched.sum()}/{len(kmh)} frames matched; paint {100 * paint:.1f}% of the road band")
    if moving.any():
        print(f"median speed while moving (>10 km/h): {np.median(kmh[moving]):.1f} km/h")
    print(f"distance (gaps interpolated): {filled.sum() / args.hz / 3.6:.0f} m")

    destination = Path(args.out) if args.out else run / "road_speed.json"
    destination.write_text(json.dumps({
        "method": "road-surface BEV correlation, marking-weighted along-road gradient, "
                  "dark non-paint excluded, 1/2/3-frame consistency",
        "plane_pitch_deg": round(float(pitch), 3), "plane_height_m": round(float(height), 3),
        "plane_source": plane_source, "near_limit_m": round(near_m, 2), "near_source": near_source,
        "paint_share": round(paint, 3), "hz": args.hz,
        "matched_fraction": round(float(matched.mean()), 3),
        "speed_kmh": [None if not np.isfinite(v) else round(float(v), 3) for v in kmh]}))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
