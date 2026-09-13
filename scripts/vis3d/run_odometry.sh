#!/bin/bash

#SBATCH --job-name=vis3d_odometry
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --time=08:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma

# Everything ego-motion, for a whole dataset, in one place.
#
#   sbatch scripts/vis3d/run_odometry.sh                      # data/test
#   DATASET=CARE_YTB sbatch scripts/vis3d/run_odometry.sh     # all clips
#   CLIPS="close_bike four_way" bash scripts/vis3d/run_odometry.sh
#   STAGES=dash FORCE=1 bash scripts/vis3d/run_odometry.sh    # redo dashes only
#   DATASET=CARE_YTB STAGES=road,select,report FORCE=1 sbatch scripts/vis3d/run_odometry.sh
#
# STAGES (comma-separated, default "lanes,intrinsics,vo,plane,dash,road,heading,select,report"):
#
#   lanes       detect_lanes.py          -> lane_masks/
#   intrinsics  estimate_intrinsics.py   -> camera_intrinsics.json     [GPU, 'processor' env]
#   depth       infer_unidepth_on_frames -> samples-pseudodepth/       [GPU, slow, OFF by default]
#   vo          estimate_ego_motion.py   -> ego_poses.txt              [needs depth]
#               plot_ego_trajectory.py   -> ego_trajectory.png
#   plane       calibrate_plane.py       -> road_plane.json
#   dash        detect_dashes.py         -> dashes.json, dash_detections.jpg
#               dash_odometry.py         -> dash_speed.json
#               plot_dash_odometry.py    -> dash_odometry_vs_ground_truth.png
#   road        road_odometry.py         -> road_speed.json
#               speed from the road surface itself (bird's-eye correlation on the
#               car-detection plane, painted markings weighted up, shadows left
#               out). On the 11 truthed clips: 8 within 50-150% of distance;
#               fails at highway speed (changelane 31%, too_close 44%) and on
#               wet roads (close_slam 34%). road_speed.json is skipped where it
#               already exists, so after changing road_odometry.py re-run with
#               STAGES=road,select,report FORCE=1.
#   heading     plot_dash_odometry.heading_from_essential -> dash_heading.npy
#               cumulative yaw per frame from the essential matrix between
#               consecutive frames (the reliable half of visual odometry: rotation
#               without scale). generate_ras_logs.py combines it with
#               ego_speed.json to build the exported trajectory.
#   select      select_odometry.py       -> ego_speed.json
#               one speed per clip: dash if the plane was calibrated from lane
#               lines, else road. Every file carries reliability "unverified".
#   report      odometry_report.py       -> <dataset>/odometry_report.json
#
# CHECKS AND GENERATION. A stage makes what it needs instead of skipping:
#
#   needs                     made by (when missing)        who needs it
#   lane_masks/               detect_lanes.py               plane
#   camera_intrinsics.json    estimate_intrinsics.py [GPU]  plane, depth, road
#   road_plane.json           calibrate_plane.py            dash, select; road
#                                                           when there are no
#                                                           car detections
#   dash_speed.json or        road_odometry.py              select
#     road_speed.json
#
# Prerequisites are only made when missing -- FORCE regenerates a stage's OWN
# output, never the inputs it borrows. What this script cannot make is flagged
# per clip: mask_results_preds.json and drivable_masks/ come from
# run_render_vis3d.sh stage 1, and without them road odometry falls back to
# road_plane.json and loses its road corridor and bonnet edge.
#
# FRAME RATE. Every speed is metres per frame times frames per second, and the
# dash and road matchers search a fixed distance per frame, so a 2 Hz clip read as
# 10 Hz is 5x too slow and mostly unmatched. HZ=auto (the default) finds each
# clip's rate: info.json "hz" if set, else the rate at which the clip's first
# extracted frames line up with its own mp4 (from the START of the clip, so a
# clip whose frames were trimmed at the end still reads correctly), else 10 Hz
# with a warning. HZ=<number> forces one rate for every clip. Dash and road
# odometry were validated only at 10 Hz; any other rate is flagged in the log.
#
# The report always covers the WHOLE dataset, including with CLIPS=: it reads
# what is on disk, and writing it for a subset would overwrite the dataset's
# odometry_report.json with only those clips.
#
# 'depth' is off by default: hours of GPU for a point map this footage breaks
# anyway (the ego's own bonnet comes back at 25-30 m on 18 of the 50 clips), and
# data/test deliberately has no samples-pseudodepth.
set -euo pipefail
SELF=$(readlink -f "$0")          # resolved before any cd: workers re-enter by path
eval "$(conda shell.bash hook)"

BASE=/fs/nexus-projects/sim2real/aliu/RAP
DATASET=${DATASET:-test}
DATA=$BASE/data/$DATASET
GT=$BASE/data/test/osd_ground_truth.json
RUN=${RUN:-1}
HZ=${HZ:-auto}
VALIDATED_HZ=10
SAMPLES=${INTRINSIC_SAMPLES:-16}
VO_METHOD=${VO_METHOD:-pointcloud}
STAGES=${STAGES:-lanes,intrinsics,vo,plane,dash,road,heading,select,report}
FORCE=${FORCE:-}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
PARALLEL=${PARALLEL:-$(( CPUS > 2 ? CPUS / 2 : 1 ))}
HEADING_ARG=""; [ -n "${NO_HEADING:-}" ] && HEADING_ARG="--no_heading"

stage_on() { [[ ",$STAGES," == *",$1,"* ]]; }
have()     { [ -n "$FORCE" ] && return 1; [ -e "$1" ]; }                 # a stage's own output
missing()  { [ ! -e "$1" ] || { [ -d "$1" ] && [ -z "$(ls -A "$1" 2>/dev/null)" ]; }; }
note()     { echo "  $*"; }

cd "$BASE/vis3d"

# ---------------------------------------------------------------- dispatcher --
if [ "${1:-}" != "--one" ]; then
    if [ -n "${CLIPS:-}" ]; then
        TARGETS="$CLIPS"
    else
        TARGETS=$(cd "$DATA" && for d in */; do
            [ -d "${d}frames" ] && printf '%s ' "${d%/}"
        done)
    fi
    if [ -z "$TARGETS" ]; then
        echo "error: no clip directories with frames/ under $DATA" >&2
        exit 1
    fi
    echo "=== dataset : $DATA"
    echo "=== stages  : $STAGES${FORCE:+  (FORCE: regenerating the output of each stage)}"
    echo "=== clips   : $TARGETS"
    echo "=== rate    : HZ=$HZ (validated at $VALIDATED_HZ Hz)"
    echo "=== running : $PARALLEL at a time on $CPUS cpus${HEADING_ARG:+, heading pass skipped}"
    if [ "$PARALLEL" -gt 1 ]; then
        # Prefix each worker's lines, or two clips interleaving read as one clip
        # that was abandoned half way through.
        printf '%s\n' $TARGETS | xargs -P "$PARALLEL" -I{} \
            bash -c 'bash "$0" --one "$1" 2>&1 | sed "s/^/[$1] /"' "$SELF" {} || true
    else
        for clip in $TARGETS; do bash "$SELF" --one "$clip" || true; done
    fi
    # No --clips: the report reads the whole dataset from disk, so a CLIPS= run
    # refreshes its own rows without dropping everyone else's.
    if stage_on report; then
        python3 "$BASE/vis3d/odometry_report.py" --data "$DATA" --run "$RUN" --gt "$GT"
    fi
    exit 0
fi

# ------------------------------------------------------------------ one clip --
clip=$2
dir=$DATA/$clip
out=$dir/$RUN
echo
echo "================ $clip ================"
[ -d "$dir/frames" ] || { note "stop: no frames at $dir/frames"; exit 0; }
mkdir -p "$out"

# --- prerequisite generators: make an input only when it is missing -----------
# Each returns success iff the thing exists afterwards, so a caller can write
# `need_x "why" || { note ...; }`. They switch to the env their tool needs and
# back, so they can be called from anywhere below.

need_lanes() {      # need_lanes <who needs it>
    if missing "$out/lane_masks"; then
        note "no lane_masks: generating them (needed by $1)"
        conda activate vis3d
        python detect_lanes.py --frames_dir "$dir/frames" --output_dir "$out" \
            || note "WARNING: detect_lanes failed"
        conda deactivate
    fi
    ! missing "$out/lane_masks"
}

need_intrinsics() { # need_intrinsics <who needs it>
    if missing "$out/camera_intrinsics.json"; then
        note "no camera_intrinsics.json: estimating it (needed by $1; WildCamera, 'processor' env)"
        conda activate processor
        python estimate_intrinsics.py --frames_dir "$dir/frames" \
            --output_dir "$out" --samples "$SAMPLES" || note "WARNING: intrinsics failed"
        conda deactivate
    fi
    [ -f "$out/camera_intrinsics.json" ]
}

calibrate() {       # calibrate: (re)write road_plane.json; needs intrinsics and lane_masks
    need_intrinsics "the road plane" || { note "no plane: intrinsics unavailable"; return 1; }
    need_lanes "the road plane" || { note "no plane: lane masks unavailable"; return 1; }
    conda activate vis3d
    # --allow_fallback: a clip that cannot be calibrated from lane lines still
    # gets a plane -- the car detections, the lane vanishing point, or a nominal
    # level camera -- and road_plane.json's `quality` says which, so the report,
    # the plots and select_odometry.py can all tell a guess from a measurement.
    local ok=0
    python calibrate_plane.py --clip "$dir" --run "$RUN" \
        --intrinsics "$out/camera_intrinsics.json" --allow_fallback --write && ok=1
    conda deactivate
    [ "$ok" = 1 ] && [ -f "$out/road_plane.json" ]
}

need_plane() {      # need_plane <who needs it>
    if [ ! -f "$out/road_plane.json" ]; then
        note "no road_plane.json: calibrating it (needed by $1)"
        calibrate || return 1
    fi
    [ -f "$out/road_plane.json" ]
}

# --- frame rate ---------------------------------------------------------------
detect_rate() {     # echoes "<hz> <how it was decided>"
    if [ "$HZ" != auto ]; then
        echo "$HZ set by HZ"
        return 0
    fi
    conda activate vis3d
    python - "$dir" "$VALIDATED_HZ" <<'PY' || echo "$VALIDATED_HZ assumed (rate check crashed)"
import json, sys
from pathlib import Path

import numpy as np

clip, default = Path(sys.argv[1]), sys.argv[2]
info = clip / "info.json"
if info.exists():
    try:
        hz = json.loads(info.read_text()).get("hz")
    except Exception:
        hz = None
    if hz:
        print(f"{float(hz):g} from info.json")
        sys.exit(0)
try:
    import cv2
except ImportError:
    print(f"{default} assumed (no opencv to check against the video)")
    sys.exit(0)
frames = sorted(p for p in (clip / "frames").iterdir() if p.suffix.lower() in {".jpg", ".png"})
videos = sorted(clip.glob("*.mp4"))
if len(frames) < 6 or len(videos) != 1:
    print(f"{default} assumed (need one mp4 and >= 6 frames to check, found {len(videos)} and {len(frames)})")
    sys.exit(0)
cap = cv2.VideoCapture(str(videos[0]))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    print(f"{default} assumed (video frame rate unreadable)")
    sys.exit(0)

def thumb(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    # the centre only: extraction fits the video to the target size by cropping edges
    grey = grey[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    small = cv2.resize(grey, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)

def at(seconds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, image = cap.read()
    return thumb(image) if ok else None

extracted = [thumb(cv2.imread(str(frames[k]))) for k in range(1, 6)]
errors = {}
for hz in (2, 5, 10, 15, 20, 25, 30):
    if hz > fps + 0.5:
        continue
    diffs = []
    for k, frame in enumerate(extracted, start=1):
        video = at(k / hz)
        if video is None:
            break
        diffs.append(float(np.mean(np.abs(frame - video))))
    if len(diffs) == len(extracted):
        errors[hz] = float(np.mean(diffs))
if len(errors) < 2:
    print(f"{default} assumed (could not read enough of the video)")
    sys.exit(0)
ranked = sorted(errors, key=errors.get)
best, runner_up = ranked[0], ranked[1]
margin = errors[runner_up] / max(errors[best], 1e-6)
if margin < 1.15:
    # a clip that starts parked looks the same at every rate
    print(f"{default} assumed (frames match {best} and {runner_up} Hz about equally, x{margin:.2f})")
else:
    print(f"{best} matched against the video ({len(frames)} frames; next best {runner_up} Hz, x{margin:.1f} worse)")
PY
    conda deactivate
}

read -r CLIP_HZ RATE_HOW < <(detect_rate) || true
CLIP_HZ=${CLIP_HZ:-$VALIDATED_HZ}
note "frame rate: $CLIP_HZ Hz (${RATE_HOW:-no detail})"
OFF_RATE=0
awk -v a="$CLIP_HZ" -v b="$VALIDATED_HZ" 'BEGIN { exit !((a - b) ^ 2 > 0.25) }' && OFF_RATE=1
[ "$OFF_RATE" = 1 ] && note "WARNING: $clip is at $CLIP_HZ Hz, not $VALIDATED_HZ Hz -- dash and road odometry were validated only at $VALIDATED_HZ Hz; treat its speeds as unchecked"

# --- inputs this script cannot make: say so up front --------------------------
missing "$out/mask_results_preds.json" && note "note: no mask_results_preds.json (run_render_vis3d.sh stage 1) -- road odometry cannot fit the car plane or mask vehicles"
missing "$out/drivable_masks" && note "note: no drivable_masks/ (run_render_vis3d.sh stage 1) -- road odometry loses the road corridor and bonnet edge"

# --- lane masks ---------------------------------------------------------------
if stage_on lanes; then
    if have "$out/lane_masks" && ! missing "$out/lane_masks"; then
        note "lane_masks present ($(ls "$out/lane_masks" | wc -l) frames)"
    else
        conda activate vis3d
        python detect_lanes.py --frames_dir "$dir/frames" --output_dir "$out" \
            || note "WARNING: detect_lanes failed"
        conda deactivate
    fi
fi

# --- intrinsics: needs frames and nothing else, so it runs for every clip -----
if stage_on intrinsics; then
    if have "$out/camera_intrinsics.json"; then
        note "camera_intrinsics.json present: $(python3 -c "
import json;d=json.load(open('$out/camera_intrinsics.json'))
print(f\"fx {d['fx']:.0f}, {d.get('hfov_deg',0):.0f} deg hFOV, spread {d.get('focal_spread',0)*100:.1f}%\")")"
    else
        note "estimating intrinsics (WildCamera, 'processor' env)"
        conda activate processor
        python estimate_intrinsics.py --frames_dir "$dir/frames" \
            --output_dir "$out" --samples "$SAMPLES" || note "WARNING: intrinsics failed"
        conda deactivate
    fi
fi

# --- depth, off by default ----------------------------------------------------
if stage_on depth; then
    if have "$out/samples-pseudodepth"; then
        note "samples-pseudodepth present"
    elif need_intrinsics "depth"; then
        note "running UniDepth (slow)"
        conda activate vis3d
        python infer_unidepth_on_frames.py --frames_dir "$dir/frames" --output_dir "$out" \
            --intrinsics "$out/camera_intrinsics.json" || note "WARNING: depth failed"
        conda deactivate
    else
        note "skip depth: camera_intrinsics.json could not be made"
    fi
fi

conda activate vis3d

# --- baseline visual odometry + its plot --------------------------------------
if stage_on vo; then
    if have "$out/ego_poses.txt"; then
        note "ego_poses.txt present"
    elif [ -d "$out/samples-pseudodepth" ]; then
        python estimate_ego_motion.py --frames_dir "$dir/frames" --output_dir "$out" \
            --scene "$clip" --method "$VO_METHOD" || note "WARNING: estimate_ego_motion failed"
    else
        note "skip estimate_ego_motion: no samples-pseudodepth (run with STAGES=...,depth,...)"
    fi
    if [ -f "$out/ego_poses.txt" ]; then
        python plot_ego_trajectory.py --output_dir "$out" --hz "$CLIP_HZ" \
            --title "$clip/$RUN  baseline VO ($VO_METHOD)" || note "WARNING: ego_trajectory.png failed"
    fi
fi

# --- road plane ---------------------------------------------------------------
pitch=""; height=""
if stage_on plane || stage_on dash; then
    pvar="PITCH_${clip}"; hvar="HEIGHT_${clip}"
    if [ -n "${!pvar:-}" ]; then
        pitch=${!pvar}; height=${!hvar:-1.33}
        note "plane overridden: pitch $pitch deg, height $height m"
    elif have "$out/road_plane.json"; then
        note "road_plane.json present"
    elif stage_on plane; then
        calibrate || note "WARNING: road plane could not be made"
    else
        need_plane "dash odometry" || note "WARNING: road plane could not be made"
    fi
    if [ -z "$pitch" ] && [ -f "$out/road_plane.json" ]; then
        read -r pitch height quality < <(python3 -c "
import json;d=json.load(open('$out/road_plane.json'))
print(d['pitch_deg'], d['height_m'], d.get('quality','legacy'))")
        note "plane: pitch $pitch deg, height $height m, quality $quality"
        [ "$quality" = "measured" ] || note "plane quality $quality: a fallback, not a calibration"
    fi
fi

# --- dashes, speed, and the comparison plot -----------------------------------
if stage_on dash; then
    if [ -z "$pitch" ]; then
        note "skip dash odometry: no road plane"
    else
        python detect_dashes.py --clip "$dir" --run "$RUN" \
            --intrinsics "$out/camera_intrinsics.json" \
            --pitch_deg "$pitch" --height "$height" || note "WARNING: detect_dashes failed"
        if [ -f "$out/dashes.json" ]; then
            python dash_odometry.py --clip "$dir" --run "$RUN" --hz "$CLIP_HZ" \
                || note "WARNING: dash_odometry failed"
            if [ "$OFF_RATE" = 0 ] && python3 -c "
import json,sys; sys.exit(0 if '$clip' in json.load(open('$GT'))['clips'] else 1)"; then
                python plot_dash_odometry.py --clip "$dir" --run "$RUN" \
                    --intrinsics "$out/camera_intrinsics.json" --gt "$GT" --name "$clip" \
                    $HEADING_ARG || note "WARNING: plot_dash_odometry failed"
            else
                note "no comparison plot for $clip (no burned-in ground truth, or not at $VALIDATED_HZ Hz)"
            fi
        fi
    fi
fi

# --- road-surface speed -------------------------------------------------------
run_road() {
    if ! need_intrinsics "road odometry"; then
        note "skip road odometry: camera_intrinsics.json could not be made"
        return 1
    fi
    # The car-detection plane needs detections; without them road_odometry.py
    # falls back to road_plane.json, so make sure that exists.
    if missing "$out/mask_results_preds.json"; then
        need_plane "road odometry (no car detections to fit a plane from)" \
            || { note "skip road odometry: no car detections and no road plane"; return 1; }
    fi
    python road_odometry.py --clip "$dir" --run "$RUN" --hz "$CLIP_HZ" \
        --intrinsics "$out/camera_intrinsics.json" || { note "WARNING: road_odometry failed"; return 1; }
}
if stage_on road; then
    if have "$out/road_speed.json"; then
        note "road_speed.json present"
    else
        run_road || true
    fi
fi

# --- heading: cumulative yaw per frame ----------------------------------------
# The essential matrix between consecutive frames gives rotation reliably even
# where depth-based translation is wrong. Cached as dash_heading.npy -- the same
# file plot_dash_odometry.py writes -- so a clip that already has one is reused.
if stage_on heading; then
    if have "$out/dash_heading.npy"; then
        note "dash_heading.npy present"
    elif need_intrinsics "heading"; then
        rm -f "$out/dash_heading.npy"          # FORCE: the function returns a cache as-is
        python - "$dir" "$out" <<'PY' || note "WARNING: heading pass failed"
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from plot_dash_odometry import heading_from_essential

clip, run = Path(sys.argv[1]), Path(sys.argv[2])
k = json.loads((run / "camera_intrinsics.json").read_text())
intrinsics = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
yaw = heading_from_essential(clip / "frames", run, intrinsics, run / "dash_heading.npy")
print(f"  heading: {len(yaw)} frames, net {np.degrees(yaw[-1]):+.0f} deg, "
      f"range {np.degrees(yaw.min()):+.0f} to {np.degrees(yaw.max()):+.0f} deg")
PY
    else
        note "skip heading: camera_intrinsics.json could not be made"
    fi
fi

# --- one speed per clip, chosen without ground truth --------------------------
# Always re-run (cheap, JSON only): the choice must follow whatever plane, dash
# and road outputs exist now. Its rule reads road_plane.json's quality, and it
# needs at least one speed to choose, so both are made here if absent.
if stage_on select; then
    need_plane "select (its rule reads the plane quality)" \
        || note "WARNING: no road plane -- select will treat the plane as uncalibrated and pick road"
    if [ ! -f "$out/dash_speed.json" ] && [ ! -f "$out/road_speed.json" ]; then
        note "no dash_speed.json or road_speed.json: running road odometry so select has something to choose"
        run_road || true
    fi
    python select_odometry.py --clip "$dir" --run "$RUN" || note "select: nothing to choose from"
fi
conda deactivate
