#!/usr/bin/env python3
"""Individual lane dashes as metric landmarks on the road plane.

WHY THIS EXISTS

detect_lanes.py segments a lane *line* as one continuous ribbon -- on changelane
the median connected component spans 21.8 m of range even though the paint is
dashed throughout, because the segmentation bridges the gaps. A ribbon is
useless as a landmark: its centroid is pinned by wherever the range band is cut,
not carried by the ego's motion, so tracking it reports roughly zero speed.

The paint is still there in the image. So the division of labour is:

    the mask says where the lane line is,   the image says where the paint is.

Sampling the image along the ribbon's centreline gives a 1-D profile in which
dashes are bright plateaux and the gaps between them are asphalt.

WHY IT SUBTRACTS A LOCAL BACKGROUND

Thresholding the centreline's absolute brightness fails in two ways that both
occur on this footage: the far end of the band is dimmer than the near end
(vignetting plus haze), and a white vehicle straddling the line is brighter than
any paint. Both are removed by subtracting the asphalt immediately either side
of the line at the same range -- a vehicle raises the line and its surroundings
together, paint raises only the line.

WHY IT WORKS IN THE IMAGE AND NOT IN THE BIRD'S-EYE VIEW

The warp upsamples the far band enormously and smears it into vertical streaks,
which is exactly where the dashes are shortest. Detection runs at native
resolution and only the *result* -- a handful of run endpoints -- is projected
onto the road plane.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover
    raise SystemExit("detect_dashes needs opencv") from error

# Range band to detect in. Nearer than this the lane line leaves the frame side;
# further, a 4 m dash is too few pixels to separate from its neighbour.
Z_NEAR, Z_FAR = 8.0, 34.0

# Half-width of the strip averaged along the centreline, and the offset at which
# the asphalt reference is taken, both in pixels at the range concerned. They
# scale with range because a lane line is ~0.15 m wide and subtends fewer pixels
# the further away it is.
LINE_HALF_M, BACKGROUND_OFFSET_M = 0.12, 0.55

# A dash is 2-6 m of paint. Shorter is segmentation noise or a tar patch; longer
# is either a solid line or two dashes merged by motion blur, and neither can be
# matched to itself between frames.
MIN_DASH_M, MAX_DASH_M = 1.5, 7.0

# Contrast a run must clear, as a fraction of the profile's own robust spread.
# Absolute grey levels vary too much between clips to hard-code.
CONTRAST_SIGMA = 1.0

MIN_RIBBON_PX = 150


def _road_frame(intrinsics, pitch, yaw, roll):
    from calibrate_ground import _rot_x, _rot_y, _rot_z
    rotation = _rot_x(pitch) @ _rot_y(yaw) @ _rot_z(roll)
    return np.linalg.inv(intrinsics), rotation.T


def ground_from_pixels(pixels, inverse_k, rotation_t, height):
    """Back-project image pixels onto the plane `height` below the camera.

    Returns (x, z) in metres and a mask of which pixels hit the plane in front.
    """
    rays = inverse_k @ np.c_[pixels, np.ones(len(pixels))].T
    point = rotation_t @ rays
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = height / point[1]
    ok = np.isfinite(scale) & (scale > 0)
    return (point[0] * scale), (point[2] * scale), ok


def pixels_from_ground(x, z, intrinsics, rotation, height):
    """The inverse of ground_from_pixels, for one array of ground points."""
    pts = np.stack([x, np.full_like(x, height), z], -1)
    uv = (intrinsics @ (pts @ rotation.T).T).T
    return uv[:, :2] / uv[:, 2:3]


def _ribbons(mask):
    """Connected lane-line components, largest first."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    out = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < MIN_RIBBON_PX:
            continue
        out.append(np.argwhere(labels == i)[:, ::-1])      # (u, v)
    return sorted(out, key=len, reverse=True)


def _centreline(ribbon, rows):
    """Mean column of the ribbon at each sampled row, or nan where it is absent."""
    order = np.argsort(ribbon[:, 1])
    v = ribbon[order, 1]
    u = ribbon[order, 0]
    first = np.searchsorted(v, rows, "left")
    last = np.searchsorted(v, rows, "right")
    out = np.full(len(rows), np.nan)
    for i, (a, b) in enumerate(zip(first, last)):
        if b > a:
            out[i] = u[a:b].mean()
    return out


def _sample(gray, columns, rows, half_px, offset_px):
    """Centreline brightness minus the asphalt either side, per row."""
    height, width = gray.shape
    signal = np.full(len(rows), np.nan)
    for i, (u, v, half, offset) in enumerate(zip(columns, rows, half_px, offset_px)):
        if not np.isfinite(u):
            continue
        half = max(1, int(round(half)))
        lo, hi = int(u) - half, int(u) + half + 1
        if lo < 0 or hi > width:
            continue
        line = gray[v, lo:hi].mean()
        left = int(u - offset) - half, int(u - offset) + half + 1
        right = int(u + offset) - half, int(u + offset) + half + 1
        sides = []
        for a, b in (left, right):
            if 0 <= a and b <= width:
                sides.append(gray[v, a:b].mean())
        if not sides:
            continue
        signal[i] = line - np.median(sides)
    return signal


def detect(gray, lane_mask, intrinsics, rotation, height, static_mask=None):
    """Dashes in one frame as a list of (x, z_near, z_far, z_centre, contrast).

    `static_mask` is the usual not-a-vehicle mask; a dash is road paint, so a
    detection on a moving object is always wrong.
    """
    inverse_k, rotation_t = np.linalg.inv(intrinsics), rotation.T
    image_h, image_w = gray.shape

    # Rows to sample: those a road point in the band projects to, on the frame's
    # centre column. Sampling in rows rather than in metres keeps the detection
    # at native resolution.
    probe_z = np.arange(Z_FAR, Z_NEAR, -0.1)
    probe = pixels_from_ground(np.zeros_like(probe_z), probe_z, intrinsics, rotation, height)
    rows = np.unique(np.clip(probe[:, 1].round().astype(int), 0, image_h - 1))
    if len(rows) < 12:
        return []
    # Range of each sampled row, used to size the strips and to convert back.
    row_x, row_z, ok = ground_from_pixels(
        np.c_[np.full(len(rows), intrinsics[0, 2]), rows].astype(float),
        inverse_k, rotation_t, height)
    row_z = np.where(ok, row_z, np.nan)
    scale_px = intrinsics[0, 0] / np.maximum(row_z, 1e-6)       # px per metre at that range
    half_px = np.clip(LINE_HALF_M * scale_px, 1, 25)
    offset_px = np.clip(BACKGROUND_OFFSET_M * scale_px, 3, 90)

    out = []
    for ribbon in _ribbons(lane_mask):
        columns = _centreline(ribbon, rows)
        if np.isfinite(columns).sum() < 12:
            continue
        signal = _sample(gray.astype(np.float32), columns, rows, half_px, offset_px)
        good = np.isfinite(signal)
        if good.sum() < 12:
            continue
        # Robust spread of this ribbon's own profile sets the threshold.
        centre = np.median(signal[good])
        spread = 1.4826 * np.median(np.abs(signal[good] - centre)) + 1e-6
        lit = good & (signal > centre + CONTRAST_SIGMA * spread)
        if not lit.any():
            continue
        index = np.flatnonzero(lit)
        for run in np.split(index, np.flatnonzero(np.diff(index) > 2) + 1):
            if len(run) < 2:
                continue
            z_far, z_near = row_z[run[0]], row_z[run[-1]]
            if not (np.isfinite(z_far) and np.isfinite(z_near)):
                continue
            length = z_far - z_near
            if not (MIN_DASH_M <= length <= MAX_DASH_M):
                continue
            mid = run[len(run) // 2]
            if static_mask is not None:
                u, v = columns[mid], rows[mid]
                if np.isfinite(u) and not static_mask[int(v), int(np.clip(u, 0, image_w - 1))]:
                    continue
            x, z, hit = ground_from_pixels(
                np.array([[columns[mid], rows[mid]]]), inverse_k, rotation_t, height)
            if not hit[0]:
                continue
            out.append((float(x[0]), float(z_near), float(z_far), float(z[0]),
                        float(signal[run].mean() / spread)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Detect individual lane dashes on the road plane.")
    parser.add_argument("--clip", required=True, help="clip dir holding frames/ and the run dir")
    parser.add_argument("--run", default="1")
    parser.add_argument("--intrinsics", default=None, help="camera_intrinsics.json (default: in the run dir)")
    parser.add_argument("--pitch_deg", type=float, default=None, help="override the lane-VP pitch")
    parser.add_argument("--height", type=float, default=1.33)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out", default=None, help="where to write dashes.json (default: run dir)")
    parser.add_argument("--overlay", type=int, default=3, metavar="N",
                        help="also write dash_detections.jpg showing the detections on N frames "
                             "(0 disables). A dash count is not interpretable on its own -- this "
                             "is how you see whether it is reading paint or a kerb shadow.")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_ground as CG

    clip = Path(args.clip).resolve()
    run = clip / args.run
    kpath = Path(args.intrinsics) if args.intrinsics else run / "camera_intrinsics.json"
    k = json.loads(Path(kpath).read_text())
    intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    shape = (k["height"], k["width"])

    calibration = CG.estimate(run / "lane_masks", intrinsics, shape, height=args.height)
    pitch = calibration.pitch if args.pitch_deg is None else np.radians(args.pitch_deg)
    rotation = CG._rot_x(pitch) @ CG._rot_y(calibration.yaw) @ CG._rot_z(calibration.roll)
    print(f"pitch {np.degrees(pitch):+.2f} deg, yaw {np.degrees(calibration.yaw):+.2f}, "
          f"height {args.height} m")

    names = sorted(p.name for p in (run / "lane_masks").iterdir() if p.suffix == ".png")
    names = names[args.start:args.end]
    frames_dir = clip / "frames"
    result = {}
    for name in names:
        stem = Path(name).stem
        image = frames_dir / f"{stem}.jpg"
        if not image.exists():
            continue
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(run / "lane_masks" / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            continue
        result[stem] = detect(gray, mask, intrinsics, rotation, args.height)

    counts = np.array([len(v) for v in result.values()])
    print(f"{len(result)} frames, {counts.sum()} dashes, median {np.median(counts):.0f}/frame, "
          f"{(counts > 0).mean()*100:.0f}% of frames have one")
    destination = Path(args.out) if args.out else run / "dashes.json"
    destination.write_text(json.dumps(
        {"pitch_deg": float(np.degrees(pitch)), "yaw_deg": float(np.degrees(calibration.yaw)),
         "height_m": args.height, "frames": result}))
    print(f"wrote {destination}")

    if args.overlay > 0:
        overlay = _overlay(result, frames_dir, intrinsics, rotation, args.height, args.overlay)
        if overlay is not None:
            path = run / "dash_detections.jpg"
            cv2.imwrite(str(path), overlay)
            print(f"wrote {path}")


def _overlay(result, frames_dir, intrinsics, rotation, height, count):
    """Detections drawn back onto the frames they came from.

    Worth the few lines: "231 dashes, median 1/frame" is the same summary whether
    the detector is reading lane paint or the shadow of a kerb, and only the
    picture separates them.
    """
    stems = [s for s in sorted(result) if result[s]][:max(count * 4, count)]
    if not stems:
        return None
    step = max(1, len(stems) // count)
    panes = []
    for stem in stems[::step][:count]:
        image = cv2.imread(str(Path(frames_dir) / f"{stem}.jpg"))
        if image is None:
            continue
        for x, z_near, z_far, z_centre, _ in result[stem]:
            zs = np.linspace(z_near, z_far, 24)
            pixels = pixels_from_ground(np.full_like(zs, x), zs, intrinsics, rotation, height)
            cv2.polylines(image, [pixels.astype(int).reshape(-1, 1, 2)], False, (0, 255, 255), 5)
            cv2.putText(image, f"{z_centre:.0f}m", (int(pixels[0, 0]) + 12, int(pixels[0, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image, f"{stem}  ({len(result[stem])} dashes)", (24, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        panes.append(image[int(image.shape[0] * 0.40):int(image.shape[0] * 0.94)])
    if not panes:
        return None
    stacked = np.vstack(panes)
    return cv2.resize(stacked, (stacked.shape[1] // 2, stacked.shape[0] // 2))


if __name__ == "__main__":
    main()
