#!/bin/bash
# Submits one sbatch job per clip, each running run_gemini.sh over that clip's
# own "event_frames" window -- or, for a clip whose info.json has no such key,
# over every frame it has. Run this on the login node -- it only submits and
# exits in a second. The #SBATCH directives live in run_gemini.sh, which is what
# each job actually runs; there are deliberately none here.
#
#   ./run_gemini_batch.sh                        # submit every clip in CLIPS
#   LIST=1 ./run_gemini_batch.sh                 # print the plan and cost, submit nothing
#   CLIPS="ambulance blocker" ./run_gemini_batch.sh
#   CONCURRENT=1 ./run_gemini_batch.sh           # all at once -- see the warning below
#
# Every run_gemini.sh knob is forwarded to all jobs (--export=ALL carries the
# submitting environment): STRIDE, OPS, PROPAGATE, MODEL, RPM, APPLY, FROM_CACHE,
# IN_BOXES.
#
# SOURCE in the plan says which boxes each clip is reviewed FROM. A clip that
# already has gemini_boxes_3d.json is reviewed as it now stands ("gemini") and
# corrected in place, so a second pass sees the first pass's edits -- and any
# made by hand since -- rather than re-judging the raw lift and re-reporting
# faults that have already been fixed. A clip without one falls back to the lift
# ("lift"). IN_BOXES=boxes_3d.json forces the raw lift everywhere ("forced").
#
#   PROPAGATE=track ./run_gemini_batch.sh
#   FROM_CACHE=1 PROPAGATE=track ./run_gemini_batch.sh   # re-apply, no requests
#
# THIS SPENDS REAL MONEY, roughly $0.006 a frame. LIST=1 first.

set -uo pipefail          # not -e: one clip failing to submit must not stop the rest

BASE=/fs/nexus-projects/sim2real/aliu/RAP
DATASET="${DATASET:-CARE_YTB}"
LIST="${LIST:-0}"
# Jobs are chained by default: each waits on the one before it (afterany, so a
# failure still lets the next start). The Gemini rate limit is per ACCOUNT, not
# per job -- N clips running at once share one --rpm and spend the batch
# retrying 429s. Set CONCURRENT=1 to submit them independently, and divide RPM
# by the number of clips if you do.
CONCURRENT="1"

# The Gemini rate limit is per ACCOUNT, so it is a budget shared by every job in
# flight, not a per-job allowance. Set it here once: RPM_TOTAL is the whole
# account's requests/minute, and each job gets RPM_TOTAL divided by however many
# run at the same time.
#
#   CONCURRENT=1   N jobs at once    -> each job gets RPM_TOTAL / N
#   CONCURRENT=0   one job at a time -> that job gets the whole budget
#
# Your real limit is per-account and is shown in the AI Studio dashboard rather
# than the public docs -- check there and set this to match. 15 is a
# conservative paid-tier value; the free tier was 5/min.
RPM_TOTAL="60"
# Setting RPM directly overrides all of this and is used verbatim.

# The clips to review, in order. Override with CLIPS="a b c".
CLIPS=(
    # "barrier_whip  Remove the red box on top and add box for the road barrier."
    # "bike_crosswalk  Keep the boxes corresponding to the biker and the car that crashed separate."
    # "buick_nearmiss  Enlarge the box for the white car that almost hit."

    # "changelane  Fix box sizing for black car that was collided into."
    # "close_bike  Shrink bike box to better fit and remove bottom box on the dashboard."
    # "closetruck  Add and fix box for the truck that was nearly hit."
    # "crash_back  Fix box size and orientation for the car that was hit from behind."
    # "deer_family  Add boxes for each deer and remove the box on the dashboard."
    # "deercross  Add box for the deer that ran across the highway."
    # "fallen_load  Add boxes for the yellow roll that fell off the truck and the motorcylist that was hit."
    # "fast_close  Fix box size for the black car that was nearly hit."

    # "flying_tire  Add boxes for the tire that flew off the truck and patch missing boxes for the car in front that was hit."
    # "grandma_crash  Fix box sizing for the truck that was hit and remove the box on the dashboard."
    # "help  Fix box size for the black car that was nearly hit and patch missing box when close."
    # "highway_hazard  Add box for the cone on highway and remove the box on the dashboard."
    # "incoming  Fix box size for the car that was nearly hit and patch missing box when close."
    # "merge_in  Lengthen the box for the nearby car to include the trailor and remove the box on the dashboard."
    # "motorcycle_fall  Add boxes for the motorcycle and motorcyclist that fell and remove the box on the dashboard."
    # "night_deer  Fix box size for the deer that was nearly hit and patch missing box when close."
    # "opposing_crash  Fix box sizes for the cars that crashed and remove the box on the dashboard."
    # "petty  Adjust orientation of truck in front and remove box in dashboard."
    # "poland_slip  Fix box size for pedestrian that was nearly hit and patch missing boxes after stop."
    # "pull_out  Fix box sizes for the cars nearby"
    # "reverse_roundabout  Fix box sizes for incoming truck and remove box in dashboard."
    # "roundabout  Fix orientation for all boxes in each frame and remove box in dashboard."
    # "street_race  Adjust box size for car that was hit and patch missing box during collision."
    # "turn_overtake  Increase box sizes for the cars on other side of the road."
    # "uturn  Fix box size for truck that was hit and patch missing box during collision."
    # "yield_runway  Fix box size for the car that was nearly hit and patch missing boxes when passing."
    "grandma_crash  Fix box on truck to better fit and patch missing boxes."
)

# Extra per-clip instructions, layered on the prompt in run_gemini.sh: one line
# per note under a [clip] heading, optionally prefixed "frames A-B:" to scope it
# (which also forces those frames into the review). Clips with no section are
# reviewed on the prompt alone.
NOTES_FILE="${NOTES_FILE:-}"
[ -f "$NOTES_FILE" ] || NOTES_FILE=""

# Frames this clip will cost, and how that number was arrived at: "<n> window"
# for an event_frames range, "<n> all" for a clip with no window, which
# run_gemini.sh reviews end to end. Must agree with run_gemini.sh's own
# resolution or the bill printed here is not the bill charged -- both take the
# frame numbers off the filenames and slice by STRIDE.
frames_for() {
    python3 -c '
import json, sys
from pathlib import Path
info, frames_dir, stride = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
try:
    window = json.loads(info.read_text()).get("event_frames")
except FileNotFoundError:
    window = None
if window:
    print(len(range(window[0], window[1] + 1, stride)), "window"); sys.exit()
if not frames_dir.is_dir():
    print(-1, "-"); sys.exit()
n = len(sorted(p for p in frames_dir.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[::stride])
print(n if n else -1, "all")
' "$BASE/data/$DATASET/$1/info.json" "$BASE/data/$DATASET/$1/frames" "${STRIDE:-1}" \
      2>/dev/null || echo "-1 -"
}

printf '%-20s %5s %-6s %8s %-7s  %s\n' CLIP FRAMES '' COST SOURCE REQUEST
total=0; runnable=(); requests=()
for entry in "${CLIPS[@]}"; do
    clip="${entry%% *}"                                  # first word
    note="${entry#"$clip"}"                              # whatever follows it
    note="${note#"${note%%[![:space:]]*}"}"              # trim leading spaces
    read -r n scope < <(frames_for "$clip")
    src=$( [ -f "$BASE/data/$DATASET/$clip/1/${OUT_BOXES:-gemini_boxes_3d.json}" ] \
           && echo gemini || echo lift )
    [ -n "${IN_BOXES:-}" ] && src="forced"
    if [ "$n" -lt 0 ]; then
        printf '%-20s %8s %10s   %s\n' "$clip" "-" "-" \
            "SKIP: no event_frames and no frames/ -- is $DATASET/$clip a real clip?"
        continue
    fi
    shown="$note"
    [ ${#shown} -gt 46 ] && shown="${shown:0:43}..."
    printf '%-20s %5s %-6s %8s %-7s  %s\n' "$clip" "$n" "$scope" \
        "\$$(python3 -c "print(f'{$n*0.0057:.2f}')")" "$src" "${shown:--}"
    total=$((total + n)); runnable+=("$clip"); requests+=("$note")
done
echo
echo "$(( ${#runnable[@]} )) clip(s), $total frames, ~\$$(python3 -c "print(f'{$total*0.0057:.2f}')")"
if [ -n "${RPM:-}" ]; then
    per_job_rpm="$RPM"; rpm_why="RPM set explicitly"
elif [ "$CONCURRENT" = 1 ] && [ ${#runnable[@]} -gt 1 ]; then
    per_job_rpm=$(( RPM_TOTAL / ${#runnable[@]} ))
    [ "$per_job_rpm" -lt 1 ] && per_job_rpm=1
    rpm_why="$RPM_TOTAL/min split across ${#runnable[@]} concurrent jobs"
else
    per_job_rpm="$RPM_TOTAL"; rpm_why="whole $RPM_TOTAL/min budget, one job at a time"
fi
echo "submission: $([ "$CONCURRENT" = 1 ] && echo "concurrent, ${#runnable[@]} at once" || echo 'chained, one clip at a time')"
echo "rate:       ${per_job_rpm}/min per job  ($rpm_why)"
[ -n "$NOTES_FILE" ] && echo "notes: $NOTES_FILE" || echo "notes: none"

if [ "$LIST" = 1 ]; then
    echo; echo "LIST=1, nothing submitted."
    exit 0
fi

echo
prev=""; submitted=()
for i in "${!runnable[@]}"; do
    clip="${runnable[$i]}"; note="${requests[$i]}"
    dep=()
    [ "$CONCURRENT" != 1 ] && [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
    jid=$(sbatch --parsable \
            --job-name="gemini_$clip" \
            "${dep[@]}" \
            --export=ALL,VIDEO="$clip",DATASET="$DATASET",NOTE="$note",RPM="$per_job_rpm",NOTES_FILE="$NOTES_FILE",IN_BOXES="${IN_BOXES:-}" \
            "$BASE/scripts/vis3d/run_gemini.sh")
    if [ -z "$jid" ]; then
        echo "!! $clip: sbatch failed, skipping" >&2
        continue
    fi
    printf '  submitted %-20s job %-10s %s%s\n' "$clip" "$jid" \
        "$([ ${#dep[@]} -gt 0 ] && echo "after $prev " || echo "")" \
        "$([ -n "$note" ] && echo "[request]" || echo "")"
    submitted+=("$jid"); prev="$jid"
done

echo
echo "${#submitted[@]} job(s) submitted. Watch them with:"
echo "  squeue -u \$USER --name=$(printf 'gemini_%s,' "${runnable[@]}" | sed 's/,$//')"
echo "Logs: $BASE/my_dump/gemini_<clip>.out.<jobid>"
