#!/bin/bash
#SBATCH --job-name=gemini
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --time=10:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma

set -euo pipefail

BASE=/fs/nexus-projects/sim2real/aliu/RAP

# The key lives in gemini_key.sh, which is gitignored so this script can be
# committed without it. An absolute path, NOT one relative to BASH_SOURCE:
# sbatch copies the batch script to a spool directory on the compute node, so
# under Slurm BASH_SOURCE points at /var/spool/... and the key is not beside it.
KEY_FILE="$BASE/scripts/vis3d/gemini_key.sh"
if [ ! -f "$KEY_FILE" ]; then
    echo "error: $KEY_FILE not found." >&2
    echo "       Create it with:  echo 'export GEMINI_API_KEY=...' > $KEY_FILE" >&2
    exit 1
fi
source "$KEY_FILE"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set by $KEY_FILE}"

############################### CONFIG ###############################
DATASET="${DATASET:-test}"
VIDEO="${VIDEO:-close_nightcrash}"
RUN="${RUN:-1}"
STRIDE="${STRIDE:-1}"         # 1 = review every frame of the window
FRAMES="${FRAMES:-}"

FIND="${FIND:-}"              # extra classes to hunt; empty = review what is there
# Extra instructions layered on top of the prompt, for a fault you already know
# about. A note may start "frames 40-90:" to scope it, which also forces those
# frames into the review whatever the window says.
#   NOTE='frames 85-95: the crossing car is boxed at half its length'
NOTE="${NOTE:-}"              # one note for this clip
NOTES_FILE="${NOTES_FILE:-}"  # a file of notes, one per line, grouped under
                              # [clip] headings -- the batch-friendly form
OPS="${OPS:-fix,delete,add}"  # drop "add" to keep a pass purely corrective
BOXES_FROM="${BOXES_FROM:-fit}"
# Which boxes file to review. Empty = auto: a clip that already has
# gemini_boxes_3d.json is reviewed as it now stands, so a second pass sees the
# first pass's corrections (and any hand edits made since) instead of re-judging
# boxes that have already been fixed -- and re-reporting faults that are gone.
# A clip with no such file falls back to the lift's own boxes_3d.json.
# Set IN_BOXES=boxes_3d.json to force a review of the raw lift.
IN_BOXES="${IN_BOXES:-}"
OUT_BOXES="${OUT_BOXES:-gemini_boxes_3d.json}"
# frame = edit only the frames actually reviewed. With STRIDE=1 every frame of
#         the window is judged first-hand, so a box the reviewer deliberately
#         left alone stays alone -- and nothing outside the window is touched.
# track = push each correction along the object's whole track, across all 155
#         frames. Fills gaps when the stride is coarse, but overwrites a
#         reviewed frame's own verdict with its nearest neighbour's.
PROPAGATE="${PROPAGATE:-frame}"
MAX_ADD_GAP="${MAX_ADD_GAP:-10}"   # only used when PROPAGATE=track
MODEL="${MODEL:-gemini-3.8-flash}"
RPM="${RPM:-3}"               # requests/min; the batch script computes this per job
# Frames in flight at once. The throttle sets the pace only if a worker can
# finish a request inside the interval: with WORKERS=1 the ceiling is one
# request per round trip (~3-8 s for a 1080p frame), so raising RPM past ~10
# does nothing on its own. 4 is the Python's own default.
WORKERS="${WORKERS:-4}"
APPLY="${APPLY:-1}"           # 1 = write gemini_boxes_3d.json beside boxes_3d.json;
DRY_RUN="${DRY_RUN:-0}"
FROM_CACHE="${FROM_CACHE:-0}" # re-derive from findings.json, costing no requests
######################################################################

# --- the review rules ------------------------------------------------------
# Everything Gemini is told about the task. A general pass: refit what sits
# badly, delete what is not a road user, leave correct boxes alone.
read -r -d '' PROMPT <<'PROMPT_END' || true
You are reviewing 3D object boxes that an automatic pipeline fitted to dashcam
footage. Each box is drawn on the frame as a coloured wireframe cuboid carrying
its number, like #3. The four thick edges of a wireframe are the FRONT face of
the object the box claims to be on -- the end a vehicle drives towards.

This is a general quality pass. Look at every wireframe in the frame, and also
at every road user that has no wireframe at all. Report what is wrong in one of
three ways. A wireframe that sits on its object is correct: say nothing about
it.

1. FIX a box that is on a real object but does not sit on it properly -- shifted
   off it, too big, too small, or facing the wrong way.

   Return the box's number and box_2d: the 2D bounding box the object ACTUALLY
   occupies in this frame, as [ymin, xmin, ymax, xmax] normalised to 0-1000.
   The 3D box is re-measured from that, so box_2d is the whole correction. Draw
   it tightly around the visible object, as you would if you were labelling the
   object from scratch -- not around the wireframe, and not around both.

   Your box_2d must still overlap the wireframe you are correcting. A target
   that misses it entirely reads as two wireframe numbers swapped, and is
   dropped rather than applied. Correct a box onto the object it is already
   nearest; never move it onto a different object.

   If, and only if, the box faces the wrong way, set `heading` to one of
   same_as_ego, oncoming, crossing_left_to_right, crossing_right_to_left. The
   commonest error is a reversed heading: the thick FRONT face drawn on the back
   of a car driving away, or on the front of one that is oncoming. When the
   direction is already right, leave `heading` out or set it to "unchanged" --
   most fixes are position and size only.

2. DELETE a box that is not on a real road user at all.

   The commonest case by far is the ego vehicle itself. The car this camera is
   mounted on fills the bottom of every frame, and a wireframe drawn over its
   own bonnet, wipers, dashboard, windscreen pillars or the reflection in its
   glass is always wrong -- there is no object there. Boxes on buildings,
   hedges, road signs, poles, shadows and empty tarmac are wrong the same way.

   Delete rather than fix when nothing is there to box. Fix is for a box that
   has found a real object and merely sits on it badly.

3. ADD a box for a road user that has none.

   Every vehicle, pedestrian, cyclist or animal on or beside the road should
   carry a wireframe. If one has no box on it, add it -- however small or
   distant, as long as you can tell what it is. One finding per object: three
   pedestrians standing together is three findings with three separate box_2d,
   never one box around the group.

   Only add where there is genuinely no wireframe. If the object already has a
   box that merely sits badly, that is a FIX, not an add -- an add that lands on
   top of an existing box is discarded as a duplicate.

   Set `name` to the closest of: car, truck, bus, van, pedestrian, bicycle,
   motorcycle, deer, dog, horse, cow. These are the classes whose real
   proportions are known, and the 3D box is sized from them; anything else is
   treated as a car and comes out car-shaped. If an object of interest fits none
   of them, prefer the nearest by size and shape.

   Set `heading` from the way the object faces in this frame: same_as_ego,
   oncoming, crossing_left_to_right or crossing_right_to_left. For an add this
   matters more than for a fix -- heading is the one thing about a new box that
   its box_2d cannot recover, and leaving it out sits the box square to the
   road. A pedestrian crossing is not facing along it.

   Do not add boxes for things that are not road users: parked bicycles on a
   rack, vehicles inside showroom windows, images of cars on billboards or van
   liveries, or reflections in glass.

CONSISTENCY ACROSS FRAMES

Each correction you make is replayed along the whole track of the object it
lands on, not just the frame you saw it in, and boxes you add on nearby frames
are joined into one track with the frames between them interpolated. So treat the same object the same
way every time it appears: if you refit a car in one frame and leave the same
car alone in the next, the two disagree and the box will move about between
them. Judge each object as you would across the whole clip -- a box that is
consistently a little large is one steady correction, not a different one per
frame.

The corollary: do not chase small per-frame wobble. A box that breathes or
shifts slightly from frame to frame is the pipeline's noise, and refitting it
one frame at a time makes the motion worse rather than smoother. Report a box
only when it is wrong in a way you would describe the same on any frame where
you can see the object.

WHAT NOT TO REPORT

Do not report boxes that overlap each other or are drawn over one another --
that is just how the picture renders, not a fault. Do not comment on class
labels or on whether an object is interesting. Do not report a box that is
merely imperfect if it already sits on its object.

Never judge a box by its size or distance in metres. You are not shown those
numbers, and they are on a scale of their own -- a box that sits on its object
on screen is correct however implausible its metres would be.

confidence is 0-1: how sure you are this is a real mistake worth an edit.
reason is one short clause, e.g. "box sits left of the white van", "wireframe on
the ego bonnet", "front face drawn on the rear of a car driving away", or
"unboxed cyclist at the right kerb".
PROMPT_END

# --- run -------------------------------------------------------------------
# A clip with no "event_frames" window is reviewed in full rather than skipped:
# some faults -- a box on the ego bonnet, wipers or windscreen -- are not events
# and appear wherever the pipeline drew one, so there is no window to scope them
# to. Whole-clip is several times the frames of a typical window, so it is
# several times the money: run_gemini_batch.sh marks these rows "all" and prices
# them, and LIST=1 shows the bill before anything is submitted.
if [ -z "$FRAMES" ]; then
    FRAMES=$(python3 -c '
import json, sys
from pathlib import Path
info, frames_dir, stride = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
try:
    window = json.loads(info.read_text()).get("event_frames")
except FileNotFoundError:
    window = None                      # no info.json reads as no window
if window:
    lo, hi = window
    frames = list(range(lo, hi + 1, stride))
else:
    if not frames_dir.is_dir():
        sys.exit("no \"event_frames\" in %s and no frames at %s -- add the window, "
                 "or set FRAMES= explicitly" % (info, frames_dir))
    # Numbers come off the filenames, not a 0..n-1 range: a clip trimmed with
    # --keep-numbering, or one with holes, still gets frames that exist.
    frames = sorted(int(p.name.split(".")[0]) for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[::stride]
    if not frames:
        sys.exit("no frames at " + str(frames_dir))
print(",".join(str(f) for f in frames))
' "$BASE/data/$DATASET/$VIDEO/info.json" "$BASE/data/$DATASET/$VIDEO/frames" "$STRIDE") || exit 1
fi

# `conda activate` in a non-interactive shell needs the hook first, which is what
# "Run 'conda init' before 'conda activate'" is actually complaining about.
# set +u around it: conda's own activation scripts read unset variables.
eval "$(conda shell.bash hook)"
set +u
conda activate vis3d
set -u

PROMPT_FILE=$(mktemp "${TMPDIR:-/tmp}/gemini_review_prompt.XXXXXX")
trap 'rm -f "$PROMPT_FILE"' EXIT
printf '%s\n' "$PROMPT" > "$PROMPT_FILE"

# Resolve IN_BOXES here rather than in the Python so the choice is printed with
# the run and shows up in the job log.
RUN_DIR="$BASE/data/$DATASET/$VIDEO${RUN:+/$RUN}"
if [ -z "$IN_BOXES" ]; then
    if [ -f "$RUN_DIR/$OUT_BOXES" ]; then
        IN_BOXES="$OUT_BOXES"
    else
        IN_BOXES=boxes_3d.json
    fi
fi
echo "reviewing $IN_BOXES -> $OUT_BOXES in $RUN_DIR"
# `if`, not `[ ... ] && echo`: under set -e a false test there is the whole
# statement's exit status, so the common case -- a clip with no reviewed file,
# where the two names differ -- would end the run right here.
if [ "$IN_BOXES" = "$OUT_BOXES" ]; then
    echo "  (in place: $OUT_BOXES is both the source and the target)"
fi

FLAGS=(--dataset "$DATASET" --clip "$VIDEO" --run "$RUN"
       --frames "$FRAMES" --ops "$OPS" --max_add_gap "$MAX_ADD_GAP"
       --propagate "$PROPAGATE"
       --model "$MODEL" --rpm "$RPM" --workers "$WORKERS" --boxes_from "$BOXES_FROM"
       --in_boxes "$IN_BOXES" --out_boxes "$OUT_BOXES"
       --system_file "$PROMPT_FILE")
for class in $FIND; do FLAGS+=(--find "$class"); done
if [ -n "$NOTE" ]; then FLAGS+=(--note "$NOTE"); fi
if [ -n "$NOTES_FILE" ]; then FLAGS+=(--notes_file "$NOTES_FILE"); fi
# if, not `[ ... ] && FLAGS+=(...)`: a false test there is the last command of
# the list, so under set -e a run with these at 0 would exit right here.
if [ "$APPLY" = 1 ]; then FLAGS+=(--apply); fi
if [ "$DRY_RUN" = 1 ]; then FLAGS+=(--dry-run); fi
if [ "$FROM_CACHE" = 1 ]; then FLAGS+=(--from_findings); fi

cd "$BASE/vis3d"
python -u gemini_review_boxes.py "${FLAGS[@]}"
