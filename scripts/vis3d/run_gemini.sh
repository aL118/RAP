#!/bin/bash
# Review deer_family's boxes with Gemini and apply what it finds.
#
#   ./run_gemini.sh                    # ask, then write into boxes_3d.json
#   DRY_RUN=1 ./run_gemini.sh          # ask, write nothing
#   FROM_CACHE=1 ./run_gemini.sh       # re-derive from the last run's answers, no requests
#
# The review rules live in this file, in the PROMPT heredoc below. Edit them
# here rather than in gemini_review_boxes.py: what counts as a fault is the part
# that gets tuned, and a run should sit in the same file as the instructions it
# ran under. --find / --note below still steer an individual run on top of them.

set -euo pipefail

# The key lives in gemini_key.sh, which is gitignored so this script can be
# committed without it. Resolved relative to this file, not the caller's cwd.
KEY_FILE="$(dirname "${BASH_SOURCE[0]}")/gemini_key.sh"
if [ ! -f "$KEY_FILE" ]; then
    echo "error: $KEY_FILE not found." >&2
    echo "       Create it with:  echo 'export GEMINI_API_KEY=...' > $KEY_FILE" >&2
    exit 1
fi
source "$KEY_FILE"
: "${GEMINI_API_KEY:?GEMINI_API_KEY not set by $KEY_FILE}"

############################### CONFIG ###############################
DATASET="${DATASET:-test}"
VIDEO="${VIDEO:-deer_family}"
RUN="${RUN:-1}"

# Every 8th frame from 40 to 160. Eight rather than every frame because the free
# tier allows 20 requests a day and 40-161 is 122 frames; eight rather than
# twenty because MAX_ADD_GAP below is 10, and two sightings further apart than
# that are left as separate boxes instead of being joined into a track. So this
# is the coarsest stride that still fills the gaps.
FRAMES="${FRAMES:-40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160}"

FIND="${FIND:-deer}"          # classes to box whether or not the detector knows them
OPS="${OPS:-add,delete}"      # what the run is allowed to do at all
# model = Gemini returns the 3D box itself (box3d) and it is written as given.
# fit   = Gemini returns a 2D box and the 3D one is measured from it by
#         back-projecting onto the road. See --boxes_from in the Python.
BOXES_FROM="${BOXES_FROM:-model}"
MAX_ADD_GAP="${MAX_ADD_GAP:-10}"   # furthest two adds may be and still be interpolated
MODEL="${MODEL:-gemini-3.7-flash}"
RPM="${RPM:-3}"               # free tier is 5/min and 20/day; 3 leaves headroom
APPLY="${APPLY:-1}"           # 1 = merge into boxes_3d.json (reversible via .lifted.json)
DRY_RUN="${DRY_RUN:-0}"
FROM_CACHE="${FROM_CACHE:-0}" # re-derive from findings.json, costing no requests
######################################################################

# --- the review rules ------------------------------------------------------
# Everything Gemini is told about the task. Three rules for this clip: box the
# deer, delete the boxes on the ego bonnet, touch nothing else.
read -r -d '' PROMPT <<'PROMPT_END' || true
You are reviewing 3D object boxes that an automatic pipeline fitted to dashcam
footage. Each box is drawn on the frame as a coloured wireframe cuboid carrying
its number, like #3. The four thick edges of a wireframe are the FRONT face of
the object the box claims to be on -- the end a vehicle drives towards.

You have exactly two jobs on this footage. Do both, and nothing else.

1. ADD a box for every deer that is on or beside the road and has no wireframe
   on it. One finding per animal -- three deer standing together is three
   findings with three separate box_2d, never one box around the group. Box them
   however small and distant they are, as long as you can tell it is a deer.
   Set `name` to "deer".

2. DELETE any box that is not on a real road user. On this clip that is above
   all the ego vehicle's own bonnet: the car this camera is mounted on fills the
   bottom of every frame, and a wireframe drawn over that shiny dark hood is
   always wrong. Boxes on buildings, hedges, shadows and empty tarmac are wrong
   the same way.

Do NOT do anything else. Do not report a box as misplaced, mis-sized or
mis-headed; do not comment on class labels; do not report boxes that overlap
each other or are drawn over one another, which is just how the picture renders.
A correct wireframe is one you say nothing about.

Never judge a box by its size or distance in metres. You are not shown those
numbers, and they are on a scale of their own -- a box that sits on its object
on screen is correct however implausible its metres would be.

For every add and every delete you may also give box_2d -- the object's 2D
bounding box as [ymin, xmin, ymax, xmax] normalised to 0-1000. It is not used to
build the 3D box; it is recorded so a person can see which animal you meant.

For every add, also set `heading` from the way the animal faces in this frame:
crossing_left_to_right, crossing_right_to_left, oncoming, or same_as_ego. A deer
crossing a road is not facing along it, and heading is the one thing about an
added box that its box_2d cannot recover -- leave it out and the box sits square
to the road.

confidence is 0-1: how sure you are this is a real mistake worth an edit.
reason is one short clause, e.g. "deer standing on right shoulder" or "box on
the ego bonnet".

WHAT YOUR ANSWERS BECOME

Your findings are assembled into a file called gemini_boxes_3d.json: a complete
copy of the pipeline's own boxes_3d.json with your corrections merged in, so the
original is never modified and the two can be rendered side by side. Its shape
is one entry per frame, keyed by frame filename, each holding parallel lists --
"boxes", "names", "scores" -- plus the camera. A delete drops one entry from all
three lists; an add appends one to each. A box in "boxes" looks exactly like
this:

  {"x": 32.99, "y": 8.17, "z": 0.81,
   "length": 3.58, "width": 1.43, "height": 1.19,
   "roll": 0.0, "pitch": 0.0, "yaw": -0.11}

x, y, z are the box centre in the ego frame in metres: x forward from the
camera, y to the LEFT, z up, with z = 0 at the road. length, width and height
are its extent in metres. roll, pitch and yaw are radians, yaw about the
vertical axis and counter-clockwise seen from above -- so 0 faces the way the
ego car does, +pi/2 faces left, -pi/2 faces right.

For every ADD, return this box yourself, as `box3d`, with x, y, z, length,
width, height and yaw. roll and pitch are always 0 and you do not need to send
them. These numbers go into the file as you give them -- nothing recomputes
them -- so they are the whole answer, not a hint.

To place one, work from what the frame shows you:

  - Range (x) is the hard part. Use the road to judge it. Lane markings, the
    dashes of a centre line, kerbs, poles and driveways all recede at a known
    rate, and an animal standing level with a feature is at that feature's
    distance. A deer whose feet are near the vanishing point is far; one filling
    a third of the frame height is close.
  - y is how far LEFT of the camera's own axis it stands, negative to the right.
    An animal on the right shoulder of a two-lane road is roughly -4 to -6.
  - z is the centre height above the road, so about half the animal's height for
    something standing on it -- around 0.7 for a deer.
  - length, width and height are the animal itself: an adult deer is roughly
    1.8 long, 0.6 wide, 1.4 tall, a fawn noticeably smaller. Use what you see.
  - yaw follows the heading you reported: 0 facing the way the ego car does,
    +1.57 facing left, -1.57 facing right, 3.14 facing back towards the camera.

Be careful and consistent between frames of the same animal: a deer walking
across a road changes y quickly and x slowly, and never jumps twenty metres
between one frame and the next.
PROMPT_END

# --- run -------------------------------------------------------------------
BASE=/fs/nexus-projects/sim2real/aliu/RAP

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

FLAGS=(--dataset "$DATASET" --clip "$VIDEO" --run "$RUN"
       --frames "$FRAMES" --ops "$OPS" --max_add_gap "$MAX_ADD_GAP"
       --model "$MODEL" --rpm "$RPM" --workers 1 --boxes_from "$BOXES_FROM"
       --system_file "$PROMPT_FILE")
for class in $FIND; do FLAGS+=(--find "$class"); done
# if, not `[ ... ] && FLAGS+=(...)`: a false test there is the last command of
# the list, so under set -e a run with these at 0 would exit right here.
if [ "$APPLY" = 1 ]; then FLAGS+=(--apply); fi
if [ "$DRY_RUN" = 1 ]; then FLAGS+=(--dry-run); fi
if [ "$FROM_CACHE" = 1 ]; then FLAGS+=(--from_findings); fi

cd "$BASE/vis3d"
python -u gemini_review_boxes.py "${FLAGS[@]}"
