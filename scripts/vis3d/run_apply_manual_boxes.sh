#!/bin/bash
# Merge hand-drawn boxes into a clip's stage-1 output, on a GPU.
#
#   ./run_apply_manual_boxes.sh                               # the CONFIG below
#   VIDEO=closetruck ./run_apply_manual_boxes.sh
#   VIDEO="closetruck ice_road" ./run_apply_manual_boxes.sh   # several clips
#   DRY_RUN=1 ./run_apply_manual_boxes.sh                     # report, write nothing
#
# Run it directly -- it submits itself with sbatch and returns, so the work
# keeps going after you close your laptop. There are no #SBATCH directives:
# every resource is passed on the sbatch command line below, which is what lets
# --output be a variable rather than a constant baked into a comment.
#
# The same merge runs automatically as stage 1c inside run_render_vis3d.sh, so
# this exists for the case where that is not what you want: seeing what the
# boxes will do -- how much of each one SAM actually fills, how many detector
# boxes they suppress -- before spending a re-render on them. On the login node
# the ViT-H encoder takes minutes per annotated frame, which is what makes this
# worth a job rather than a foreground command.
#
# This only rewrites mask_results_preds.json. boxes_3d.json and everything drawn
# from it stay as they were until stage 3 runs again -- set VIDEO/RUN in
# rerun_vis3d.sh and submit that, which re-lifts and re-rasterizes. Stage 1c
# there will re-run this merge itself, which costs nothing and is idempotent.

############################### CONFIG ###############################
DATASET="${DATASET:-CARE_YTB}"   # dataset dir under data/
VIDEO="${VIDEO:-closetruck}"     # clip, or several separated by spaces
RUN="${RUN-1}"                   # run subdir holding the clip's outputs ("" = clip dir)

MANUAL_BOX_IOU="${MANUAL_BOX_IOU:-0.5}"   # detector box overlapping a manual one by this much is dropped
# On the frames a drawn track does NOT cover, a detection overlapping where the
# track just was is given that track's id, so a detector that caught the object
# only in glimpses has those glimpses carried by the annotation instead of
# deleted by MIN_TRACK_LEN. ADOPT_GAP=0 turns it off.
MANUAL_ADOPT_IOU="${MANUAL_ADOPT_IOU:-0.3}"   # overlap with the track's last box that claims a detection
MANUAL_ADOPT_GAP="${MANUAL_ADOPT_GAP:-5}"     # frames a track may go unseen before its last box is too stale
# 1 = prompt SAM with each drawn box for a real mask. 0 = use the filled
# rectangle, which needs no GPU at all -- but hands the lifter every bright
# pixel of background inside the box, and on the footage that needs hand-drawn
# boxes in the first place (blown-out, backlit) that is most of it.
USE_SAM="${USE_SAM:-1}"
DRY_RUN="${DRY_RUN:-0}"          # 1 = report what would happen, write nothing

# One fixed file, overwritten every run, rather than %x.out.%j: this job prints
# a dozen lines that are worth reading once -- mask fill, how many detector
# boxes were suppressed -- and my_dump already holds 300-odd logs from jobs
# whose output was worth keeping. Point SLURM_LOG at a %j path on a run that is.
# Two of these submitted at once would write over each other, so submit a list
# in one job (VIDEO="a b c") rather than a job each.
SLURM_LOG="${SLURM_LOG:-/fs/nexus-projects/sim2real/aliu/RAP/my_dump/dump}"

# SAM ViT-H is a 2.4 GB checkpoint and about 8 GB of GPU memory once the image
# encoder is resident; 16 GB of host memory is room to load it and hold one
# 1920x1080 frame at a time. The work is a handful of encoder passes, so the
# wall clock is dominated by loading the model off /fs.
SBATCH_ARGS=(
    --job-name=apply_manual_boxes
    --mem=16gb --gres=gpu:1 --ntasks=1 --cpus-per-task=4
    --time=00:30:00 --qos=default --account=gamma --partition=gamma
)
######################################################################

set -euo pipefail

BASE=/fs/nexus-projects/sim2real/aliu/RAP

# --- submit half ------------------------------------------------------------
# Everything above runs twice: once here, once on the node. SLURM_JOB_ID is the
# only thing that tells the two apart, and slurm sets it for us.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    # Checked here, before submitting, rather than on the node: a clip that does
    # not exist or was never annotated is the likeliest thing to go wrong, and
    # it belongs in the terminal you are standing at, not in a log you would
    # have to know to go and read.
    for clip in $VIDEO; do
        clip_dir="$BASE/data/$DATASET/$clip"
        out_dir="$clip_dir${RUN:+/$RUN}"
        [ -d "$clip_dir" ] || { echo "error: no such clip: $clip_dir" >&2; exit 1; }
        [ -e "$out_dir/manual_boxes.json" ] || {
            echo "error: no manual_boxes.json in $out_dir" >&2
            echo "       draw some first:  python vis3d/annotate_boxes.py --clip $clip \\" >&2
            echo "                             --dataset $DATASET --run $RUN" >&2
            exit 1; }
        [ -e "$out_dir/mask_results_preds.json" ] || {
            echo "error: no mask_results_preds.json in $out_dir -- run stage 1 first" >&2
            exit 1; }
    done

    mkdir -p "$(dirname "$SLURM_LOG")"
    echo "submitting: $VIDEO ($DATASET${RUN:+, run $RUN})"
    echo "output:     $SLURM_LOG"
    # --export=ALL so the CONFIG above reaches the node as *this* run set it,
    # not as the defaults the copy on the node would compute for itself.
    exec sbatch "${SBATCH_ARGS[@]}" --output="$SLURM_LOG" --error="$SLURM_LOG" \
        --export=ALL "$0"
fi

# --- job half ---------------------------------------------------------------
eval "$(conda shell.bash hook)"
# set +u around the activation: conda's own activation scripts read unset
# variables, which is fatal under set -u. Same reason as run_render_vis3d.sh.
set +u
conda activate vis3d
set -u

cd "$BASE/vis3d"

# if, not `[ ... ] && FLAGS+=(...)`: a false test there is the last command of
# the list, so under set -e a run with DRY_RUN=0 would exit right here.
FLAGS=()
if [ "$USE_SAM" != 1 ]; then FLAGS+=(--no-sam); fi
if [ "$DRY_RUN" = 1 ]; then FLAGS+=(--dry-run); fi

# SAM is rebuilt per clip rather than once for all of them. On a GPU the load is
# most of the runtime, so a long list pays it repeatedly -- but a clip that fails
# then takes only itself down and leaves the others' output untouched, which for
# hand-drawn boxes is the trade worth making.
status=0
for clip in $VIDEO; do
    out_dir="$BASE/data/$DATASET/$clip${RUN:+/$RUN}"
    echo "== $clip ($DATASET${RUN:+, run $RUN})"
    if ! python apply_manual_boxes.py --output_dir "$out_dir" \
             --frames_dir "$BASE/data/$DATASET/$clip/frames" \
             --iou "$MANUAL_BOX_IOU" \
             --adopt_iou "$MANUAL_ADOPT_IOU" --adopt_gap "$MANUAL_ADOPT_GAP" \
             "${FLAGS[@]}"; then
        echo "FAIL $clip" >&2
        status=1
    fi
    echo
done

if [ "$status" = 0 ] && [ "$DRY_RUN" != 1 ]; then
    echo "Merged. boxes_3d.json is now stale -- re-lift with rerun_vis3d.sh"
    echo "(DATASET=$DATASET VIDEO=$VIDEO RUN=$RUN, RUN_STAGE3_LIFT=1) to see it."
fi
exit $status
