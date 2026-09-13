#!/usr/bin/env python3
"""One ego speed per clip: dash odometry or road odometry, chosen without ground truth.

    python select_odometry.py --clip ../data/CARE_YTB/turn_blocker

THE RULE

Use dash odometry when the clip's road plane was calibrated from lane lines
(road_plane.json quality `measured`, `approximate`, or the pre-quality
`legacy`); otherwise -- a plane fitted to car detections, the lane vanishing
point, or a nominal guess -- use road odometry.

WHY PLANE QUALITY, AND NOT THE ESTIMATORS' OWN MATCH RATES

A plane calibrates from lane lines only when the lane masks are following real
paint, which is exactly what dash detection needs; when calibration fails, the
"dashes" are unreliable too. Road odometry needs no paint and uses the car plane
it was validated on. On the 11 ground-truthed CARE clips, with the first road
odometry, this picked the better estimator on all 8 where either was within
~25% of true distance:

    dash  ambulance 91%, changelane 83%, close_bike 99%, too_close 86%
    road  reserved_lane 114%, turn_blocker 101%, turn_overtake 103%, yield_runway 112%

RECHECK after road_odometry.py became marking-weighted (v4, 2026-09-12): its
distances are turn_blocker 101%, turn_overtake 106%, yield_runway 109%,
reserved_lane 133%, four_way 150%, exit_now 54%, close_slam 34%. The rule now
picks the worse estimator on four_way (dash -118% on a legacy plane, road 150%)
and close_slam (road 34%, dash 70%); the other nine are unchanged.

A match-rate threshold (dash if it matched >= 40% of frames) got ambulance
wrong: dash matched only 24% of its frames and was still the accurate one.

WHAT THIS DOES NOT DO

It does not say whether the chosen speed is trustworthy. The three clips where
both estimators fail -- close_slam 57%, exit_now 57%, four_way negative -- are
not caught by anything available without ground truth that has been tested:
match rate, dash/road agreement, and a calibration-free optical-flow reference
(its rank correlation with true speed is negative on two of the 11 clips) all
fail to separate them from the good picks. `reliability` is therefore written
as "unverified" on every clip. A real flag needs a larger ground-truth set.
"""
import argparse
import json
from pathlib import Path

DASH_PLANES = {"measured", "approximate", "legacy"}


def choose(run: Path):
    """(source name, speed file, plane quality, reason)."""
    plane = run / "road_plane.json"
    quality = json.loads(plane.read_text()).get("quality", "legacy") if plane.exists() else "none"
    dash, road = run / "dash_speed.json", run / "road_speed.json"
    if quality in DASH_PLANES and dash.exists():
        return "dash", dash, quality, f"plane calibrated from lane lines ({quality})"
    if road.exists():
        why = (f"plane not calibrated from lane lines ({quality})" if quality not in DASH_PLANES
               else "lane-calibrated plane but no dash_speed.json")
        return "road", road, quality, why
    if dash.exists():
        return "dash", dash, quality, f"no road_speed.json; dash is all there is ({quality})"
    return None, None, quality, "neither dash_speed.json nor road_speed.json exists"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run", default="1")
    parser.add_argument("--out", default=None, help="default: <run>/ego_speed.json")
    args = parser.parse_args()

    run = Path(args.clip).resolve() / args.run
    source, path, quality, reason = choose(run)
    if source is None:
        raise SystemExit(f"no speed to select: {reason}")
    speed = json.loads(path.read_text())
    values = speed["speed_kmh"]
    matched = sum(v is not None for v in values) / max(len(values), 1)
    destination = Path(args.out) if args.out else run / "ego_speed.json"
    destination.write_text(json.dumps({
        "source": source, "source_file": path.name, "plane_quality": quality,
        "rule": "dash if road_plane quality in {measured, approximate, legacy}, else road",
        "reason": reason, "reliability": "unverified",
        "hz": speed.get("hz", 10.0), "matched_fraction": round(matched, 3),
        "speed_kmh": values}))
    print(f"chose {source} ({reason}); {100 * matched:.0f}% of frames have a speed; "
          f"reliability unverified -- wrote {destination}")


if __name__ == "__main__":
    main()
