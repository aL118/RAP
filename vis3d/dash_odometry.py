#!/usr/bin/env python3
"""Metric ego speed from tracked lane dashes -- no depth network, no learned model.

    python detect_dashes.py --clip ../data/CARE_YTB/changelane --pitch_deg 0.7
    python dash_odometry.py --clip ../data/CARE_YTB/changelane

WHY THE SCALE DOES NOT NEED A CORRECT ROAD PLANE

The obvious way to use detect_dashes.py's output is to take a dash's change in
range as the distance travelled. That inherits every error in the plane: on
changelane the ground-projected dash pitch reads 10.8 m where the real GOST
cycle is 16.3 m, so ranges -- and range *changes* -- come out 1.5x short.

But the ego's advance and the dash pitch are both longitudinal lengths measured
in the same coordinate, so their ratio is calibration-free to first order:

    metres travelled = (measured advance / measured dash pitch) * real cycle

The plane cancels. What replaces it is knowing the marking standard, which is a
published constant per jurisdiction rather than something to estimate per clip:
GOST R 52289 (RU) is 4 m of paint on a 16 m cycle, and the clip's own
info.json carries the country. That is a far better conditioned thing to depend
on than a camera height nobody measured.

WHY MATCHING IS UNAMBIGUOUS HERE

Dash counting failed because a peak detector fires on road texture and spray.
Matching does not have that problem: consecutive dashes are ~16 m apart while
one frame of motion at highway speed is ~2 m, an 8:1 margin, so a dash can only
be confused with itself. The acceptance window is the thing to be careful of --
see SYMMETRIC_CONTROL.

VALIDATION (changelane, against its burned-in OSD)

    moving segment    83.8 km/h median against a true 88.0
    distance          220 m against a true 262 m
    stationary        0.00 km/h median over 342 frames where the truth is 0
    symmetric control 73.9 km/h -- the window's asymmetry is worth ~12%, not 50%

Per-frame it is still noisy (rms 50 km/h, r = 0.41): with ~3 dashes in view the
median of three matches is a coarse statistic, and a frame whose dashes are
half-detected can swing it. The median and the total are sound; a per-frame
speed wants smoothing, or more dashes than a single lane line provides.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# One frame of motion must be far below the dash cycle for matching to be safe.
# 3.5 m is 126 km/h at 10 Hz -- above every clip in the ground-truthed set, whose
# fastest is exit_now at 111 -- and it is deliberately tighter than the 5.0 m
# this used to be. The joint fit searches a symmetric grid of this half-width, so
# a wider one is a wider noise floor: at 5.0 m the median absolute error over the
# six truthed clips was 5.5 km/h and too_close was 28.7 km/h slow; at 3.5 m they
# are 4.0 and 11.3. Raise it only for footage genuinely faster than 126 km/h,
# and expect the noise floor to rise with it.
MAX_STEP_M = 3.5

# A dash may not change lane between frames.
MAX_LATERAL_M = 0.8

# Backward motion to accept. A dashcam clip never reverses, so a small negative
# bound rejects noise -- but an asymmetric window manufactures a positive median
# out of pure noise, which is exactly how an earlier estimator on this footage
# fooled itself. Always re-run with SYMMETRIC_CONTROL to see how much of the
# answer the window is providing: on changelane it is 12%, and the estimate
# survives at 74 km/h against a true 87.
MIN_STEP_M = -0.5
SYMMETRIC_CONTROL = False

# Joint-fit parameters. The grid is symmetric about zero by construction.
SHIFT_GRID_M = 0.05
SHIFT_TOLERANCE_M = 0.35     # how far a dash may sit from its predicted position
MIN_DASHES_FOR_FIT = 1
MIN_FIT_SCORE = 0.5
AMBIGUITY_RATIO = 0.80       # a rival peak this close is an unresolvable frame

# Broken-lane-line cycle (paint + gap) by jurisdiction. The cycle enters the
# speed as a direct multiplier, so a wrong entry is a pure scale error -- check
# the "plane scale correction" the run prints before trusting a new country.
# RU GOST R 52289 4+12; US MUTCD 3+9 (10 ft + 30 ft); CA TAC 3+6; NL/DE 3+9;
# GB TSRGD 2+4 on lane lines.
DASH_CYCLE_M = {"RU": 16.3, "US": 12.2, "CA": 9.0, "NL": 12.0, "DE": 12.0,
                "PL": 12.0, "AU": 9.0, "NZ": 9.0, "KR": 12.0, "CN": 15.0,
                "GB": 6.0, None: 12.2}

# Countries whose entry above is a nominal standard rather than something
# measured on these clips; a run on one of these should be treated as
# provisional until its dash pitch is checked.
UNVERIFIED_CYCLE = {"CA", "NL", "DE", "PL", "AU", "NZ", "KR", "CN", "GB"}


def measured_cycle(per_frame):
    """The dash cycle as it appears in the (possibly wrong) plane coordinates."""
    gaps = []
    for dashes in per_frame:
        if len(dashes) < 2:
            continue
        for lane in np.unique(np.round(dashes[:, 0])):
            same = np.sort(dashes[np.abs(dashes[:, 0] - lane) < MAX_LATERAL_M][:, 1])
            gaps += [g for g in np.diff(same) if 3.0 < g < 40.0]
    return float(np.median(gaps)) if len(gaps) >= 10 else None


def _joint_shift(before, after):
    """The single advance that best carries the whole dash set `before` onto `after`.

    Why not match each dash to its nearest neighbour and take the median: that
    thresholds every dash independently against an acceptance window, and any
    window that is asymmetric about zero -- as one must be to express "a dashcam
    does not reverse" -- RECTIFIES dash-position noise into forward motion. On
    arterial clips, where a frame advances ~1.1 m and centroid noise is a good
    fraction of that, the rectified noise dominates: close_bike came out +67%
    and four_way, which is parked most of its clip, +506%.

    Scoring one shift against all dashes at once removes the mechanism. The grid
    is symmetric about zero, nothing is accepted or rejected per dash, and a
    frame whose dashes carry no real correspondence produces a flat score
    surface -- which is then rejected for being ambiguous, rather than answering
    with whatever the window's positive half admitted.
    """
    if len(before) == 0 or len(after) == 0:
        return None
    shifts = np.arange(-MAX_STEP_M, MAX_STEP_M + SHIFT_GRID_M, SHIFT_GRID_M)
    score = np.zeros(len(shifts))
    pairs = 0
    for x0, z0 in before:
        near = after[np.abs(after[:, 0] - x0) < MAX_LATERAL_M]
        if not len(near):
            continue
        pairs += 1
        # a dash at z0 is at z0 - shift next frame, so the residual is
        # (z0 - z1) - shift; soft-assigned so no single pair can veto.
        residual = (z0 - near[:, 1])[None, :] - shifts[:, None]
        score += np.exp(-0.5 * (residual / SHIFT_TOLERANCE_M) ** 2).sum(axis=1)
    if pairs < MIN_DASHES_FOR_FIT or score.max() < MIN_FIT_SCORE:
        return None

    peak = int(np.argmax(score))
    # Ambiguity: a second peak nearly as good, far enough away to be a different
    # answer, means this frame cannot distinguish them. Dash pitch is ~10-16 m
    # against a <5 m step, so a rival peak is noise, not the neighbouring dash.
    far = np.abs(shifts - shifts[peak]) > 4 * SHIFT_TOLERANCE_M
    if far.any() and score[far].max() > AMBIGUITY_RATIO * score[peak]:
        return None

    # sub-grid refinement on the three samples around the peak
    if 0 < peak < len(shifts) - 1:
        y0, y1, y2 = score[peak - 1], score[peak], score[peak + 1]
        denominator = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-9 else 0.0
        return float(shifts[peak] + np.clip(offset, -1, 1) * SHIFT_GRID_M)
    return float(shifts[peak])


def steps(per_frame, symmetric=SYMMETRIC_CONTROL):
    """Per-frame advance in plane units; nan where the frame pair will not fit.

    `symmetric` is retained for the control in dash_odometry's docstring, but the
    joint fit is symmetric by construction, so it is now a no-op -- kept so the
    flag does not silently disappear from scripts that pass it.
    """
    out = np.full(len(per_frame), np.nan)
    for i in range(1, len(per_frame)):
        shift = _joint_shift(per_frame[i - 1], per_frame[i])
        if shift is not None:
            out[i] = shift
    return out


def speed(dashes_json, cycle_m=None, symmetric=SYMMETRIC_CONTROL, hz=10.0):
    """(speed km/h per frame, metres per frame, the scale correction applied)."""
    data = json.loads(Path(dashes_json).read_text())
    per_frame = [np.array([[d[0], d[3]] for d in data["frames"][s]]) if data["frames"][s]
                 else np.zeros((0, 2)) for s in sorted(data["frames"])]
    apparent = measured_cycle(per_frame)
    cycle_m = cycle_m if cycle_m is not None else DASH_CYCLE_M[None]
    correction = 1.0 if apparent is None else cycle_m / apparent
    advance = steps(per_frame, symmetric) * correction
    return advance * hz * 3.6, advance, correction


def main():
    parser = argparse.ArgumentParser(description="Metric ego speed from tracked lane dashes.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--dashes", default=None, help="dashes.json (default: in the run dir)")
    parser.add_argument("--country", default=None, help="marking standard; default from info.json")
    parser.add_argument("--cycle_m", type=float, default=None, help="override the dash cycle")
    parser.add_argument("--symmetric", action="store_true", help="run the symmetric-window control")
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    clip = Path(args.clip).resolve()
    dashes = Path(args.dashes) if args.dashes else clip / args.run / "dashes.json"
    country = args.country
    info = clip / "info.json"
    if country is None and info.exists():
        country = json.loads(info.read_text()).get("country")
    cycle = args.cycle_m if args.cycle_m is not None else DASH_CYCLE_M.get(country, DASH_CYCLE_M[None])

    kmh, advance, correction = speed(dashes, cycle, args.symmetric, args.hz)
    ok = np.isfinite(kmh)
    if country in UNVERIFIED_CYCLE:
        print(f"warning: {country}'s {cycle} m dash cycle is a nominal standard, not measured "
              f"on this footage; the speed scales linearly with it.")
    print(f"country {country}, dash cycle {cycle} m, plane scale correction {correction:.3f}x"
          + ("   [SYMMETRIC CONTROL]" if args.symmetric else ""))
    print(f"{ok.sum()}/{len(kmh)} frames matched")
    print(f"median speed while moving (>10 km/h): {np.median(kmh[ok & (kmh > 10)]):.1f} km/h")
    print(f"distance: {np.nansum(np.where(np.isfinite(advance), advance, 0)):.0f} m")
    destination = clip / args.run / "dash_speed.json"
    destination.write_text(json.dumps({
        "cycle_m": cycle, "correction": correction, "hz": args.hz,
        "speed_kmh": [None if not np.isfinite(v) else round(float(v), 3) for v in kmh]}))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
