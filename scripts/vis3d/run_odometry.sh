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
#   DATASET=CARE_YTB sbatch scripts/vis3d/run_odometry.sh     # all 49
#   CLIPS="close_bike four_way" bash scripts/vis3d/run_odometry.sh
#   STAGES=dash FORCE=1 bash scripts/vis3d/run_odometry.sh    # redo dashes only
#
# Replaces refit_camera.sh and trajectory_introspect.sh, which between them were
# missing four stages: neither ran detect_lanes, trajectory_introspect could not
# generate ego_poses.txt at all (it only plotted poses that already existed),
# neither persisted dash_odometry's speed, and the dash-detection overlay was
# only ever made by hand.
#
# STAGES (comma-separated, default "lanes,intrinsics,vo,plane,dash,report"):
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
#   report      a per-clip status table and odometry_report.json
#
# 'depth' is off by default because it is hours of GPU for a point map this
# footage breaks anyway (the ego's own bonnet comes back at 25-30 m on 18 of the
# 50 clips), and because data/test deliberately has no samples-pseudodepth. Turn
# it on only when you actually intend to rebuild the point-cloud VO.
#
# INTERPRETABILITY. Every stage writes a file you can look at, and the report
# says per clip which ones exist, which stage stopped it, and -- where the clip
# has burned-in ground truth -- how far out the answer is. A stage that cannot
# run says so and the clip carries on to the stages that can; nothing is skipped
# silently. Caveat worth keeping in view: dash odometry currently integrates to
# 24-84% of true distance depending on the clip, so treat its numbers as
# diagnostic rather than as training-ready labels.
set -euo pipefail
SELF=$(readlink -f "$0")          # resolved before any cd: workers re-enter by path
eval "$(conda shell.bash hook)"

BASE=/fs/nexus-projects/sim2real/aliu/RAP
DATASET=${DATASET:-test}
DATA=$BASE/data/$DATASET
GT=$BASE/data/test/osd_ground_truth.json
RUN=${RUN:-1}
HZ=${HZ:-10}
SAMPLES=${INTRINSIC_SAMPLES:-16}
VO_METHOD=${VO_METHOD:-pointcloud}
STAGES=${STAGES:-lanes,intrinsics,vo,plane,dash,report}
FORCE=${FORCE:-}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
PARALLEL=${PARALLEL:-$(( CPUS > 2 ? CPUS / 2 : 1 ))}
HEADING_ARG=""; [ -n "${NO_HEADING:-}" ] && HEADING_ARG="--no_heading"

stage_on() { [[ ",$STAGES," == *",$1,"* ]]; }
have()     { [ -n "$FORCE" ] && return 1; [ -e "$1" ]; }
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
    echo "=== stages  : $STAGES${FORCE:+  (FORCE: regenerating even where outputs exist)}"
    echo "=== clips   : $TARGETS"
    echo "=== running : $PARALLEL at a time on $CPUS cpus${HEADING_ARG:+, heading pass skipped}"
    if [ "$PARALLEL" -gt 1 ]; then
        # Prefix each worker's lines, or two clips interleaving read as one clip
        # that was abandoned half way through.
        printf '%s\n' $TARGETS | xargs -P "$PARALLEL" -I{} \
            bash -c 'bash "$0" --one "$1" 2>&1 | sed "s/^/[$1] /"' "$SELF" {} || true
    else
        for clip in $TARGETS; do bash "$SELF" --one "$clip" || true; done
    fi
    stage_on report && python3 "$BASE/vis3d/odometry_report.py" \
        --data "$DATA" --run "$RUN" --gt "$GT" --clips "$TARGETS"
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

# --- lane masks: everything geometric downstream needs them -------------------
if stage_on lanes; then
    if have "$out/lane_masks" && [ -n "$(ls -A "$out/lane_masks" 2>/dev/null)" ]; then
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
    elif [ -f "$out/camera_intrinsics.json" ]; then
        note "running UniDepth (slow)"
        conda activate vis3d
        python infer_unidepth_on_frames.py --frames_dir "$dir/frames" --output_dir "$out" \
            --intrinsics "$out/camera_intrinsics.json" || note "WARNING: depth failed"
        conda deactivate
    else
        note "skip depth: no camera_intrinsics.json"
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
        python plot_ego_trajectory.py --output_dir "$out" --hz "$HZ" \
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
        read -r pitch height < <(python3 -c "
import json;d=json.load(open('$out/road_plane.json'));print(d['pitch_deg'], d['height_m'])")
        note "road_plane.json present: pitch $pitch deg, height $height m"
    elif [ -f "$out/camera_intrinsics.json" ] && [ -d "$out/lane_masks" ]; then
        if python calibrate_plane.py --clip "$dir" --run "$RUN" \
                --intrinsics "$out/camera_intrinsics.json" --write; then
            read -r pitch height < <(python3 -c "
import json;d=json.load(open('$out/road_plane.json'));print(d['pitch_deg'], d['height_m'])")
        else
            note "stop: road plane not identifiable -- no dash odometry for this clip"
        fi
    else
        note "stop: need camera_intrinsics.json and lane_masks for the plane"
    fi
fi

# --- dashes, speed, and the comparison plot -----------------------------------
if stage_on dash && [ -n "$pitch" ]; then
    python detect_dashes.py --clip "$dir" --run "$RUN" \
        --intrinsics "$out/camera_intrinsics.json" \
        --pitch_deg "$pitch" --height "$height" || note "WARNING: detect_dashes failed"

    if [ -f "$out/dashes.json" ]; then
        python dash_odometry.py --clip "$dir" --run "$RUN" --hz "$HZ" \
            || note "WARNING: dash_odometry failed"
        if python3 -c "
import json,sys; sys.exit(0 if '$clip' in json.load(open('$GT'))['clips'] else 1)"; then
            python plot_dash_odometry.py --clip "$dir" --run "$RUN" \
                --intrinsics "$out/camera_intrinsics.json" --gt "$GT" --name "$clip" \
                $HEADING_ARG || note "WARNING: plot_dash_odometry failed"
        else
            note "no burned-in ground truth for $clip: dash_speed.json written, no comparison plot"
        fi
    fi
fi
conda deactivate
