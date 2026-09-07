#!/bin/bash
# Review a clip's 3D boxes with Gemini and write the corrections it finds.
#
#   ./run_gemini_review.sh                                  # the CONFIG below
#   VIDEO=deer_family DATASET=test FIND=deer ./run_gemini_review.sh
#   VIDEO="closetruck ice_road" ./run_gemini_review.sh      # several clips
#   VIDEO=closetruck NOTE="frames 40-90: the box on the white pickup sits left of it" \
#       ./run_gemini_review.sh
#   DRY_RUN=1 ./run_gemini_review.sh                        # report, write nothing
#
# Runs in the foreground on the login node, unlike the other runners here: this
# is all HTTP and JSON, there is no model to load and no GPU to ask for, and the
# wall clock is set entirely by the API's rate limit. A whole clip at STRIDE=5 is
# about 30 requests. Submit it with sbatch only if you are doing the whole
# dataset and want to close your laptop.
#
# Nothing is applied. It writes manual_boxes_3d.json beside boxes_3d.json, plus a
# report and the reviewed frames under gemini_review/. Read those, delete the
# edits you disagree with, then run run_apply_manual_boxes_3d or:
#
#   cd vis3d && python apply_manual_boxes_3d.py --output_dir <run dir>

############################### CONFIG ###############################
DATASET="${DATASET:-CARE_YTB}"     # dataset dir under data/
VIDEO="${VIDEO:-ambulance}"        # clip, or several separated by spaces. "" with ALL=1
RUN="${RUN-1}"                     # run subdir holding boxes_3d.json ("" = clip dir)
ALL="${ALL:-0}"                    # 1 = every clip in DATASET, ignoring VIDEO

STRIDE="${STRIDE:-5}"              # review every Nth frame
FRAMES="${FRAMES:-}"               # or exactly these, e.g. "70-110,150" (overrides STRIDE)
LIMIT="${LIMIT:-0}"                # cap frames per clip; 0 = no cap

# Faults you have already found by eye. Put in front of the reviewer for the
# frames they name, so they are confirmed and turned into edits rather than
# re-discovered. "frames 40-90: ..." also forces those frames into the review.
# NOTES_FILE holds one per line and may group them under [clip] headings.
NOTE="${NOTE:-}"
NOTES_FILE="${NOTES_FILE:-}"
# Classes to box whether or not the detector has a label for them, space
# separated: FIND="deer" on a clip whose detector only ever found cars.
FIND="${FIND:-}"

MODEL="${MODEL:-gemini-3.8-flash}"
# The free API tier allows 5 requests a minute, and above it every extra request
# is a 429 that still costs a request. Set RPM to your tier and WORKERS to 1
# under a low one -- parallelism buys nothing you are not allowed to spend.
# 4 rather than 5: the limit is enforced over a rolling window, so a run pinned
# exactly at it still trips on the drift and then waits the 40-odd seconds the
# 429 asks for -- which costs far more than the headroom does. Raise it to your
# tier's real number if you are on a paid one.
RPM="${RPM:-4}"
WORKERS="${WORKERS:-1}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.5}"

# track: one finding fixes the object on every frame of its track (default, and
# the reason a stride is affordable). frame: edit only the frames reviewed.
PROPAGATE="${PROPAGATE:-track}"
SAVE_IMAGES="${SAVE_IMAGES:-findings}"   # all | findings | none
DRY_RUN="${DRY_RUN:-0}"
######################################################################

set -euo pipefail

BASE=/fs/nexus-projects/sim2real/aliu/RAP

# Checked here rather than let the SDK fail forty frames in with a stack trace.
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "error: GEMINI_API_KEY is not set." >&2
    echo "       export GEMINI_API_KEY=... and re-run, or put it in your shell rc." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
# set +u around the activation: conda's activation scripts read unset variables,
# which is fatal under set -u. Same reason as run_render_vis3d.sh.
set +u
conda activate vis3d
set -u

FLAGS=(--dataset "$DATASET" --model "$MODEL" --rpm "$RPM" --workers "$WORKERS"
       --min_confidence "$MIN_CONFIDENCE" --propagate "$PROPAGATE"
       --save_images "$SAVE_IMAGES")
[ -n "$RUN" ] && FLAGS+=(--run "$RUN") || FLAGS+=(--run "")
if [ -n "$FRAMES" ]; then FLAGS+=(--frames "$FRAMES"); else FLAGS+=(--stride "$STRIDE"); fi
[ "$LIMIT" != 0 ] && FLAGS+=(--limit "$LIMIT")
[ -n "$NOTE" ] && FLAGS+=(--note "$NOTE")
[ -n "$NOTES_FILE" ] && FLAGS+=(--notes_file "$NOTES_FILE")
for class in $FIND; do FLAGS+=(--find "$class"); done
# if, not `[ ... ] && FLAGS+=(...)`: a false test there is the last command of
# the list, so under set -e a run with DRY_RUN=0 would exit right here.
if [ "$DRY_RUN" = 1 ]; then FLAGS+=(--dry-run); fi

if [ "$ALL" = 1 ]; then
    FLAGS+=(--all)
else
    # Checked before the first request: a clip that does not exist or was never
    # lifted belongs in the terminal you are standing at, not forty frames into
    # a rate-limited run.
    for clip in $VIDEO; do
        out_dir="$BASE/data/$DATASET/$clip${RUN:+/$RUN}"
        [ -e "$out_dir/boxes_3d.json" ] || {
            echo "error: no boxes_3d.json in $out_dir -- run stage 3 first" >&2
            exit 1; }
    done
    FLAGS+=(--clip $VIDEO)
fi

cd "$BASE/vis3d"
exec python gemini_review_boxes.py "${FLAGS[@]}"
