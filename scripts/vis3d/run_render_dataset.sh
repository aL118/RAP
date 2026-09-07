#!/bin/bash
# Take a list of video links to a rendered dataset, one job per clip.
#
# Each clip goes through three steps here: fetch the video with yt-dlp, cut it
# into frames with process_ytb.py, then sbatch run_render_vis3d.sh to do the
# expensive part. The first two are cheap and CPU-only so they run inline; only
# the third wants a GPU, and it gets one job per clip so a clip that fails takes
# only itself down and the rest keep their own GPU.
#
#   ./run_render_dataset.sh                    # process the CLIPS array below
#   ./run_render_dataset.sh blocker ice_road   # re-render these existing clips
#   ALL=1 ./run_render_dataset.sh              # render every clip on disk
#   ./run_render_dataset.sh --dir data/CARE_YTB          # every clip in a directory
#   ./run_render_dataset.sh --dir data/CARE_YTB blocker  # ... or just these
#   RERENDER=1 ./run_render_dataset.sh --dir data/CARE_YTB
#                                               # redraw vis3d/ and vis3d_overlay/
#                                               # for every clip, reusing the
#                                               # masks, depth, lanes and poses
#                                               # already on disk (no GPU)
#   DRY_RUN=1 ./run_render_dataset.sh          # print what it would do, touch nothing
#   FORCE=1 ./run_render_dataset.sh            # re-run clips that already have output
#   LOCAL=1 ./run_render_dataset.sh blocker    # run the render here instead of submitting
#   ALLOW_DUPLICATE=1 ./run_render_dataset.sh  # let a renamed url fetch a second copy
#   SUBSAMPLE_STRIDE=1 SOURCE_HZ=2 ./...        # the pre-10Hz behaviour
#   SUBSAMPLE_RUN=2hz SUBSAMPLE_FRAMES=frames_2hz FRAMES_SUBDIR=frames_10hz ./...
#                                               # also deliver a 2 Hz copy per clip
#
# DRY_RUN=1 is the check to run before a re-run: it reports, per clip, whether the
# video and frames on disk will be reused or fetched afresh, and touches nothing.
#
# Re-running with the same array is safe and is the intended way to resume: a
# clip whose video, frames or render already exist skips straight past that step.
#
# Run it directly rather than sbatch'ing it: it submits and exits, and its jobs
# are what need the GPU. Needs the vis3d env for yt-dlp, ffmpeg and node.
#
################################ INPUT ###############################
# One entry per clip, "<name>, <url>". The name is the clip's directory under
# data/$DATASET. Anything yt-dlp handles works as a url -- YouTube watch links,
# Reddit posts, Reddit /video/ links. Space around the comma is optional.
#
# A name already taken by a *different* video gets _2, _3, ... appended. A name
# already holding this same url is reused as-is, which is what makes re-running
# the array resume rather than duplicate.
# https://www.reddit.com/r/dashcams/

# wrongway, https://www.reddit.com/r/dashcams/comments/1o4294j/does_anyone_pay_attention_to_one_way_and_no_turn/
# help, https://www.reddit.com/r/dashcams/comments/comments/1kmc3cy/help/
# closetruck, https://www.reddit.com/r/dashcams/comments/16mu4d5/nearly_hit_by_18_wheeler_very_close_call_excuse/
# "grandma_crash, https://www.reddit.com/r/dashcams/comments/1w5hwko/guess_that_was_fun/"
# "street_race, https://www.reddit.com/r/dashcams/comments/1w5343a/street_race_crash_into_a_parking_lot/"
# "yield_runway, https://www.reddit.com/r/dashcams/comments/1w56j9e/that_was_close_yield_to_oncoming_through_traffic/"
# "close_bike, https://www.reddit.com/r/dashcams/comments/1vzx6y0/close_call_with_a_man_on_a_bike/"
# "opposing_crash, https://www.reddit.com/r/dashcams/comments/1w3is1d/do_you_think_this_driver_learned_their_lesson/"
# "night_deer, https://www.reddit.com/r/dashcams/comments/1w6h83k/whyd_bro_speed_up_when_he_saw_me_coming_thru_i/"
# "lamborghini, https://www.reddit.com/r/dashcams/comments/1w3iv15/lamborghini_attempts_overtake_on_a_residential/"
# "fallen_load, https://www.reddit.com/r/dashcams/comments/1w1s63x/guy_should_have_secured_his_load/"
# "close_nightcrash, https://www.youtube.com/watch?v=86YYQCMSrpY&t=12s"
# "incoming, https://www.reddit.com/r/dashcams/comments/1w4nj2c/impatient_driver/"
# "flying_tire, https://www.reddit.com/r/dashcams/comments/1vz1b2x/well_thats_a_new_driving_fear_unlocked/"
# "close_slam, https://www.reddit.com/r/dashcams/comments/1w0g23h/on_the_way_to_work/"
# "uturn, https://www.reddit.com/r/dashcams/comments/1vybip5/a_bad_driver_never_misses_their_turn/"

# "turn_blocker, https://www.reddit.com/r/dashcams/comments/1vvdyrx/a_bad_driver_never_misses_their_turn_even_if_they/" (cut out last half of frames)

# "petty, https://www.reddit.com/r/dashcams/comments/1vxsv9w/too_bad_for_him_i_feel_like_being_a_petty_asshole/"
# "ambulance, https://www.reddit.com/r/dashcams/comments/1vybyni/ambulance_nearly_creates_new_hospital_patient/"
# "too_close, https://www.reddit.com/r/dashcams/comments/1vyl8re/that_was_way_too_close/"
# "barrier_whip, https://www.reddit.com/r/dashcams/comments/1w5m9vr/always_something_new_in_dfw/"
# "one_sec, https://www.youtube.com/watch?v=86YYQCMSrpY&t=18s"
# "pull_out, https://www.reddit.com/r/dashcams/comments/1w11m1h/car_pulls_out_on_me/"
# "fast_close, https://www.reddit.com/r/dashcams/comments/1w4a6f6/well_that_was_close/"
# "roundabout, https://www.reddit.com/r/dashcams/comments/1vx305l/this_is_definitely_not_how_roundabouts_work/"
# "sacramento_crash, https://www.reddit.com/r/dashcams/comments/1vxh29g/multicar_car_crash_i5_downtown_sacramento/"
# "turn_overtake, https://www.reddit.com/r/dashcams/comments/1vx9pyz/dont_overtake_when_the_road_is_straight_do_it_on/"
# "poland_slip, https://www.reddit.com/r/dashcams/comments/1w5jx7q/quick_reactions_on_display_in_poland/"
# "keep_coming, https://www.reddit.com/r/dashcams/comments/1vwizr4/they_just_kept_coming/"
# "crash_back, https://www.reddit.com/r/dashcams/comments/1vw19b6/but_i_had_my_indicator_on_you_should_have_stopped/"
# "missing_blinker, https://www.reddit.com/r/dashcams/comments/1vwsuqn/its_not_that_hard_to_use_your_blinker_sir/"
# "exit_now, https://www.reddit.com/r/dashcams/comments/1vx1kz8/oc_i_exit_now_good_luck_everybody_else/"
# "reserved_lane, https://www.reddit.com/r/dashcams/comments/1vxneup/i_blocked_his_reserved_turn_lane/"
# "merge_in, https://www.reddit.com/r/dashcams/comments/1vvmkid/could_have_been_worse_at_80_mph/"
# "precious_space, https://www.reddit.com/r/dashcams/comments/1w1bukm/gotta_get_that_space/"
# "highway_hazard, https://www.reddit.com/r/dashcams/comments/1vvv8gf/push_bumper_vs_highway_hazard/"
# "hydroplaning, https://www.reddit.com/r/dashcams/comments/1vvtple/almost_missed_their_exit/"
# "run_red_light, https://www.reddit.com/r/dashcams/comments/1vve00k/suv_driver_runs_very_red_light/"
# "turn_blocker, https://www.reddit.com/r/dashcams/comments/1vvdyrx/a_bad_driver_never_misses_their_turn_even_if_they/"
# "four_way, https://www.reddit.com/r/dashcams/comments/1vv68is/entitled_and_unaware/"
# "deer_family, https://www.reddit.com/r/dashcams/comments/1vv08nv/something_different/"
CLIPS=()
######################################################################

BASE=/fs/nexus-projects/sim2real/aliu/RAP

# --- --dir: a directory of clips that are already on disk -------------------
# The third way in, alongside the CLIPS array and ALL=1, and the one for a
# dataset that was assembled somewhere else: point it at the directory and every
# clip in it is processed, with no urls and nothing downloaded. A clip needs
# only its own mp4 -- frames are extracted from it if they are not there yet,
# the same step the CLIPS path runs after its download.
#
# The path may be absolute or relative to $BASE, but it has to live under
# $BASE/data: run_render_vis3d.sh rebuilds each clip's path as
# $BASE/data/$DATASET/$VIDEO from the environment it is handed, so the
# directory's own name is what DATASET has to be, and a directory anywhere else
# has no name that would resolve. Given one, the check below says so rather than
# letting the jobs fail one by one on the node.
CLIP_DIR=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dir=*) CLIP_DIR="${1#--dir=}"; shift ;;
        --dir)   CLIP_DIR="${2:?--dir needs a path}"; shift 2 ;;
        --)      shift; break ;;
        *)       break ;;
    esac
done
if [ -n "$CLIP_DIR" ]; then
    CLIP_DIR="${CLIP_DIR%/}"
    case "$CLIP_DIR" in /*) ;; *) CLIP_DIR="$BASE/$CLIP_DIR" ;; esac
    if [ ! -d "$CLIP_DIR" ]; then
        echo "error: no such directory: $CLIP_DIR" >&2; exit 1
    fi
    case "$CLIP_DIR" in
        "$BASE/data/"*/*)
            echo "error: --dir must be a dataset directory directly under" >&2
            echo "       $BASE/data, not a clip inside one: $CLIP_DIR" >&2
            exit 1 ;;
        "$BASE/data/"*) ;;
        *)
            echo "error: --dir must be under $BASE/data -- the render job rebuilds" >&2
            echo "       each clip path from DATASET, so the directory's name is the" >&2
            echo "       only handle it has. Got: $CLIP_DIR" >&2
            exit 1 ;;
    esac
    DATASET="$(basename "$CLIP_DIR")"
    # One run per clip, which is what a directory assembled elsewhere has: the
    # rate-tagged default below is a leftover from when a clip held both a 10 Hz
    # and a 2 Hz copy. Overridable, and the banner prints whatever wins.
    RUN="${RUN-1}"
    # Named clips still narrow it; with none, the whole directory goes.
    [ "$#" -gt 0 ] || ALL=1
fi

############################### CONFIG ###############################
# Exported, like everything else here: sbatch --export=ALL and the LOCAL=1
# "bash $RENDER" both hand the job the *environment*, so an unexported variable
# never reaches it and the render silently falls back to its own defaults
# (DATASET=YTB, RUN=2) while the job name, expanded here, still says otherwise.
export DATASET="${DATASET:-CARE_YTB}"   # dataset dir under data/
export RUN="${RUN-10hz}"                # run subdir the pipeline writes into

# --- RERENDER: redraw from what a clip already holds -------------------------
#
# The mode for a rasterization change. Stages 1, 1b, 2 and 4 derive a clip's
# inputs -- masks, lane masks, point maps, ego poses -- and none of them look at
# the drawing code, so after a fix to the lifter, the temporal filter or the
# renderer there is nothing to recompute but stage 3. On CARE_YTB that is the
# whole difference between a re-run and a re-render: all 50 clips already carry
# every input, and stage 3 is cv2/numpy only, so the batch wants no GPU at all
# and finishes in minutes rather than holding 50 cards for hours.
#
# Three things follow from that and are set here rather than left to be
# remembered as a line of environment:
#
#   * the stages, defaulted off except the lift. Still overridable one by one --
#     RERENDER=1 RUN_STAGE2_DEPTH=1 re-infers depth and keeps the rest reused --
#     which is why each is written with :- rather than assigned outright.
#   * FORCE, because DONE_MARKER is boxes_3d.json and that is precisely the file
#     being regenerated. Without it every clip reports itself already done and
#     the batch does nothing, which reads as success.
#   * the GPU, dropped from the sbatch below unless a stage that needs one is
#     still on. See needs_gpu.
#
# What it does NOT do is check that a clip has the inputs it is about to reuse;
# submit_clip does that per clip and skips the ones that do not, so a dataset
# half-way through its first pass re-renders the finished clips and names the
# rest instead of submitting jobs that die on the node.
export RERENDER="${RERENDER:-0}"
if [ "$RERENDER" = 1 ]; then
    export RUN_STAGE1_MASKS="${RUN_STAGE1_MASKS:-0}"
    export RUN_STAGE1B_LANES="${RUN_STAGE1B_LANES:-0}"
    export RUN_STAGE2_DEPTH="${RUN_STAGE2_DEPTH:-0}"
    export RUN_STAGE3_LIFT="${RUN_STAGE3_LIFT:-1}"
    export RUN_STAGE4_VO="${RUN_STAGE4_VO:-0}"
    FORCE="${FORCE:-1}"
fi

# Fresh clips have nothing but frames/, so everything runs by default. Override
# any of these in the environment and they pass straight through to each job.
export RUN_STAGE1_MASKS="${RUN_STAGE1_MASKS:-1}"
export RUN_STAGE1B_LANES="${RUN_STAGE1B_LANES:-1}"
export RUN_STAGE2_DEPTH="${RUN_STAGE2_DEPTH:-1}"
export RUN_STAGE3_LIFT="${RUN_STAGE3_LIFT:-1}"
export RUN_STAGE4_VO="${RUN_STAGE4_VO:-1}"
export EXPORT_NAVSIM="${EXPORT_NAVSIM:-1}"
export VO_METHOD="${VO_METHOD:-openvo}"
# New clips are extracted, detected, lifted and smoothed at SOURCE_HZ, then
# delivered at SOURCE_HZ/SUBSAMPLE_STRIDE. Processing fast and delivering slow is
# strictly better than extracting slow: association gates on how far a box moves
# between frames, the temporal medians get more samples over the same stretch of
# road, and UniDepth's metric scale drifts with elapsed time rather than
# jittering per frame, so shorter links accumulate less of it. It costs
# SUBSAMPLE_STRIDE times the stage-1 and stage-2 compute and disk.
#
# SUBSAMPLE_STRIDE is the ratio between the rate the pipeline RUNS at and the
# rate run_render_vis3d.sh's frame-count windows were tuned at, so it stays at 5
# whether or not anything is delivered at the lower rate: dropping it to 1 would
# retune MAX_GAP, MAX_FILL, the two medians and the two rate
# limits to a fifth of the duration each wants. Set SOURCE_HZ=2 with it to
# reproduce a clip rendered before any of this existed.
export SOURCE_HZ="${SOURCE_HZ:-10}"        # rate the pipeline runs at
export SUBSAMPLE_STRIDE="${SUBSAMPLE_STRIDE:-5}"   # frames are this much denser than the tuning rate
# Whether a second, lower-rate copy of each finished clip is delivered as well,
# and where. Empty is no, which is the default: stage 5 then does not run, and a
# clip directory holds one frames/ and one run under their own names instead of
# <clip>/frames_10hz, <clip>/frames_2hz, <clip>/$RUN, <clip>/2hz and a pair of
# links named frames and 1 pointing at the last two. The rate-tagged frames name
# went with it -- it existed so two rates could coexist in one clip directory,
# and they no longer do.
#
# Set both to bring it back:
#   SUBSAMPLE_RUN=2hz SUBSAMPLE_FRAMES=frames_2hz FRAMES_SUBDIR=frames_10hz ./run_render_dataset.sh ...
export FRAMES_SUBDIR="${FRAMES_SUBDIR:-frames}"
export SUBSAMPLE_FRAMES="${SUBSAMPLE_FRAMES-}"
export SUBSAMPLE_RUN="${SUBSAMPLE_RUN-}"
# Every frame is fitted to this by one uniform scale plus a crop of the overflow,
# so a clip enters stage 1 at navsim's own CAM_F0 size and stays there through
# the render and the export. Never a stretch: see fit_cover in process_ytb.py.
# "native" keeps the source resolution. CROP_BIAS moves the kept strip: 0 = top,
# 0.5 = centre, 1 = bottom -- worth raising for a clip whose bonnet fills the
# bottom of the frame, since a centre crop spends half its loss on that.
export TARGET_SIZE="${TARGET_SIZE:-1920x1080}"
export CROP_BIAS="${CROP_BIAS:-0.5}"
export KEEP_DEPTH="${KEEP_DEPTH:-1}"
export SPLIT="${SPLIT:-video}"

# Video-only mp4: audio is dead weight for a frame extractor, and constraining
# the codec to avc1 keeps process_ytb.py off the VP9/AV1 paths ffmpeg is slower
# to seek in. The node runtime is yt-dlp's JS interpreter for YouTube; without
# it YouTube extraction warns and can drop formats.
YTDLP_FORMAT="bv*[ext=mp4][vcodec^=avc1]/bv*[ext=mp4]/bv*"
YTDLP_ARGS=(--js-runtimes node --remote-components ejs:github --no-playlist)

# Cookie jar, in Netscape format, for sites that gate metadata behind a login.
# Reddit is one: a post url returns 404 to a logged-out client, and neither a
# newer yt-dlp nor a browser user-agent gets round it. Export the jar from a
# browser you are signed into and point this at it. Set to "" to send none.
# YouTube links and reddit.com/video/<id> links do not need it.
COOKIES="${COOKIES-$BASE/scripts/vis3d/cookies.txt}"

# A clip is considered done when this exists under its output dir. Used only to
# skip clips on a re-run; FORCE=1 ignores it.
DONE_MARKER="boxes_3d.json"
# Which run the marker is looked for in. With a delivery run configured, the clip
# is not done until that run exists -- a high-rate pass that finished and then
# died before the subsample would otherwise be skipped as complete. With
# SUBSAMPLE_RUN empty there is no second run, and the marker is looked for in
# $RUN itself.
DONE_RUN="${DONE_RUN:-${SUBSAMPLE_RUN:-$RUN}}"

# Which stages want a card. run_render_vis3d.sh carries #SBATCH --gres=gpu:1 for
# the common case, and a command-line --gres overrides a directive in the script,
# so this is where a CPU-only batch gives its cards back. Stage 3 is cv2/numpy
# and stage 4's pointcloud method is too; only detection, lanes, depth and
# openvo need one. Getting this wrong costs nothing but a queue slot, which is
# the reason to get it right on a 50-clip batch.
SBATCH_EXTRA=()
if [ "$RUN_STAGE1_MASKS" = 1 ] || [ "$RUN_STAGE1B_LANES" = 1 ] \
   || [ "$RUN_STAGE2_DEPTH" = 1 ] \
   || { [ "$RUN_STAGE4_VO" = 1 ] && [ "$VO_METHOD" = openvo ]; }; then
    NEEDS_GPU=1
else
    NEEDS_GPU=0
    SBATCH_EXTRA+=(--gres=none)
fi

# What each enabled stage reads that it does not itself produce, as
# "<relative path>:<what it is>:<the stage that would make it>" -- checked per
# clip before submitting, so a clip missing an input is named here instead of
# failing on the node twenty minutes in. Only meaningful when the producing
# stage is off; with it on the file is about to be written.
reuse_requirements() {   # echoes one "path|label|flag" per line
    [ "$RUN_STAGE3_LIFT" = 1 ] || return 0
    [ "$RUN_STAGE1_MASKS" = 1 ] || echo "mask_results_preds.json|detections|RUN_STAGE1_MASKS"
    [ "$RUN_STAGE2_DEPTH" = 1 ] || echo "samples-pseudodepth|point maps|RUN_STAGE2_DEPTH"
    [ "$RUN_STAGE1B_LANES" = 1 ] || echo "lane_masks|lane masks|RUN_STAGE1B_LANES"
}
######################################################################

set -uo pipefail   # not -e: a failing clip is reported and stepped over, not fatal

# Absolute, not relative to $BASH_SOURCE: sbatch copies a submitted script into
# the node's spool directory, so a sibling path resolved from $BASH_SOURCE would
# point at a directory holding nothing but this file. This script only submits --
# it wants no GPU and finishes in a second -- so run it directly. It is written
# to survive being sbatch'd anyway.
RENDER="$BASE/scripts/vis3d/run_render_vis3d.sh"
EXTRACT="$BASE/vis3d/process_ytb.py"
DATA_DIR="$BASE/data/$DATASET"

[ -d "$DATA_DIR" ] || { echo "error: no such dataset directory: $DATA_DIR" >&2; exit 1; }
[ -f "$RENDER" ]   || { echo "error: no such render script: $RENDER" >&2; exit 1; }
[ -f "$EXTRACT" ]  || { echo "error: no such frame extractor: $EXTRACT" >&2; exit 1; }

# yt-dlp treats a missing --cookies file as a fatal error, so only pass the flag
# when the jar exists. A jar with no cookies for the site in hand is harmless.
if [ -n "$COOKIES" ]; then
    if [ -f "$COOKIES" ]; then
        YTDLP_ARGS+=(--cookies "$COOKIES")
    else
        echo "note: no cookie jar at $COOKIES; logged-out downloads only" >&2
    fi
fi

DRY="${DRY_RUN:-0}"

# Every clip that did not make it all the way to a submitted job, with the step
# that stopped it. Printed again at the end so one failure in a long batch is
# still visible after fifty lines of yt-dlp progress have scrolled past.
failures=()
fail() {   # fail <clip> <step> <detail>
    echo "FAIL $1 [$2]: $3" >&2
    failures+=("$1 [$2]: $3")
}

# yt-dlp asks reddit for /r/<sub>/comments/<id>/.json, which 404s when <sub> is
# not the sub the post actually lives in. A pasted link carrying the wrong
# subreddit is then indistinguishable from a deleted post, and the sub is the
# easiest part of a reddit url to get wrong. The subreddit-free /comments/<id>/
# form resolves whatever the post is filed under, so it is worth a second try.
# Longest-match on /comments/ rather than shortest, which also absorbs the
# doubled .../comments/comments/<id>/ that copying a link sometimes produces.
reddit_canonical() {   # reddit_canonical <url> -> canonical url, or nothing
    local id
    # Both forms, so the function is idempotent: feeding it a url it already
    # produced must give that url back, or comparing two urls through it can
    # never match an already-canonical one.
    case "$1" in
        *reddit.com/r/*/comments/* | *reddit.com/comments/*) ;;
        *) return 0 ;;
    esac
    id="${1##*/comments/}"
    id="${id%%/*}"
    case "$id" in
        "" | *[!a-zA-Z0-9]*) return 0 ;;   # not an id; leave the url alone
    esac
    printf 'https://www.reddit.com/comments/%s/' "$id"
}

# Strips leading and trailing whitespace, so the array can be aligned for
# reading without the padding ending up in a directory name.
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# --- name resolution --------------------------------------------------------
# Returns the directory name to use for <base,url>. A directory already holding
# this url is reused; one holding a different video pushes the name to _2, _3.
# claimed[] covers the case of two entries in the array wanting the same name in
# a single run, where neither is on disk yet at the time the other is resolved.
# The answer comes back in RESOLVED_NAME rather than on stdout: $(resolve_name)
# would run the whole thing in a subshell, and claimed[] would be discarded with
# it the moment it returned, so two array entries sharing a name would both keep
# it and the second would overwrite the first.
declare -A claimed=()
RESOLVED_NAME=""
DUPLICATE_OF=""      # set instead of RESOLVED_NAME when the guard below trips
DUPLICATE_URL=""
resolve_name() {   # resolve_name <base> <url> -> RESOLVED_NAME
    local base="$1" url="$2" cand="$1" n=1 existing
    local passed="" passed_url=""
    RESOLVED_NAME=""; DUPLICATE_OF=""; DUPLICATE_URL=""
    while [ -d "$DATA_DIR/$cand" ] || [ -n "${claimed[$cand]:-}" ]; do
        existing="${claimed[$cand]:-}"
        # A directory left behind by a failed or interrupted fetch holds nothing
        # worth keeping, so it is not a claim on the name. Without this, retrying
        # a broken link strands a fresh empty _2, _3, _4 on every attempt.
        if [ -z "${claimed[$cand]:-}" ] && [ -d "$DATA_DIR/$cand" ] \
           && [ -z "$(ls -A "$DATA_DIR/$cand" 2>/dev/null)" ]; then
            break
        fi
        if [ -z "$existing" ] && [ -f "$DATA_DIR/$cand/info.json" ]; then
            existing=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_url",""))' \
                       "$DATA_DIR/$cand/info.json" 2>/dev/null)
        fi
        # Direct match first, then both reduced to canonical form: a clip fetched
        # through the subreddit-free retry records that url while the array still
        # holds the one it was typed as, and the two must still count as one clip
        # or every re-run would strand a fresh _2.
        [ "$existing" = "$url" ] && break     # same clip, same place: resume it
        if [ -n "$existing" ]; then
            local ce cu
            ce=$(reddit_canonical "$existing"); cu=$(reddit_canonical "$url")
            [ -n "$cu" ] && [ "$ce" = "$cu" ] && break
        fi
        # The first already-fetched clip this name walks past. Only the first:
        # it is the one whose url the caller most likely meant to write.
        if [ -z "$passed" ] && [ -s "$DATA_DIR/$cand/$cand.mp4" ]; then
            passed="$cand"; passed_url="$existing"
        fi
        n=$((n + 1))
        cand="${base}_${n}"
    done

    # Stepping past a fetched clip onto a name with no video on disk means this
    # run would download and re-extract from scratch under a new name -- and the
    # commonest reason is a url edited in CLIPS after the clip was fetched (a
    # corrected subreddit, or the /video/<id> form pasted over the /comments/
    # one, which reddit_canonical cannot reduce to each other). That is a typo to
    # fix, not a second video, so it is reported rather than silently re-fetched.
    # Landing on a name that already holds its own mp4 is the ordinary resume of
    # a real _2 and passes straight through.
    if [ "${ALLOW_DUPLICATE:-0}" != 1 ] && [ -n "$passed" ] \
       && [ ! -s "$DATA_DIR/$cand/$cand.mp4" ]; then
        DUPLICATE_OF="$passed"; DUPLICATE_URL="$passed_url"
        return 0                      # RESOLVED_NAME left empty: the caller reports it
    fi

    claimed[$cand]="$url"
    RESOLVED_NAME="$cand"
}

# --- info.json --------------------------------------------------------------
# Only the two fields a machine can know are filled in. The rest are blanks to
# type over: "" for a label, [] for the event list, null for the two flags.
# null rather than false because false is a real answer: an unreviewed clip
# would read as "no glare, ego not involved" and nothing downstream could tell
# that apart from a deliberate label. Fill them in from the footage --
#   time: day | dawn_dusk | night          weather: clear | overcast | rainy | snowy
#   road_surface: dry | wet | icy | snowy  land_use: urban | suburban | rural
#   road_type: freeway | arterial | residential | parking_lot
#   event_type: [near_miss | collision | cut_in | hard_brake | ped_in_roadway | blocking]
#   sun_glare / ego_involved: true | false     country: ISO 3166-1 alpha-2
# A dashcam's own burn-in is usually the best evidence for time and country.
write_info_stub() {   # write_info_stub <clip_dir> <url>
    local info="$1/info.json"
    [ -e "$info" ] && return 0        # hand-edited labels are never overwritten
    cat > "$info" <<JSON
{
    "time": "",
    "sun_glare": null,
    "weather": "",
    "road_surface": "",
    "land_use": "",
    "road_type": "",
    "event_type": [],
    "ego_involved": null,
    "country": "",
    "source_url": "$2",
    "retrieved": "$(date +%F)"
}
JSON
}

# --- fetch + frames ---------------------------------------------------------
# Both steps are skipped when their output is already there, so a batch that
# died halfway resumes instead of re-downloading. yt-dlp only moves a file into
# place on success, so a non-empty mp4 is a complete one.
fetch_clip() {   # fetch_clip <name> <url>; echoes nothing, returns non-zero on failure
    local name="$1" url="$2" clip_dir="$DATA_DIR/$1" mp4 canon ok
    # What finally worked, which is what info.json should record: a url that
    # 404s is useless provenance. Reset only if the retry below is the one
    # that succeeded.
    local effective="$2"

    mp4="$clip_dir/$name.mp4"
    if [ -s "$mp4" ]; then
        echo "  video:  have $name.mp4"
    elif [ "$DRY" = 1 ]; then
        echo "  video:  would download $url"
    else
        mkdir -p "$clip_dir" || { fail "$name" download "cannot create $clip_dir"; return 1; }
        canon=$(reddit_canonical "$url")
        ok=0
        if yt-dlp "${YTDLP_ARGS[@]}" -f "$YTDLP_FORMAT" -o "$mp4" "$url"; then
            ok=1
        elif [ -n "$canon" ] && [ "$canon" != "$url" ]; then
            echo "  url as given failed; retrying subreddit-free: $canon"
            yt-dlp "${YTDLP_ARGS[@]}" -f "$YTDLP_FORMAT" -o "$mp4" "$canon" && ok=1
            if [ "$ok" = 1 ]; then
                effective="$canon"
                echo "  that worked -- the subreddit in the array is wrong"
            fi
        fi
        if [ "$ok" != 1 ]; then
            fail "$name" download "yt-dlp failed on $url"
            # A 404 here has two common causes and neither is visible in yt-dlp's
            # message, so name both rather than let it read as a dead link.
            case "$url" in
                *reddit.com/r/*/comments/*)
                    echo "      hint: a 404 here usually means the subreddit in the url is" >&2
                    echo "      not the one the post is in -- check it against the post. It" >&2
                    echo "      can also mean reddit wants a login: point COOKIES at a jar" >&2
                    echo "      from a signed-in browser, or use the video's own link" >&2
                    echo "      (right-click the player, Copy video URL), which needs none." >&2
                    ;;
            esac
            rmdir "$clip_dir" 2>/dev/null   # only if empty; a partial download stays
            return 1
        fi
        if [ ! -s "$mp4" ]; then
            fail "$name" download "yt-dlp wrote no $mp4"
            rmdir "$clip_dir" 2>/dev/null
            return 1
        fi
    fi

    ensure_frames "$name" || return 1

    [ "$DRY" = 1 ] || write_info_stub "$clip_dir" "$effective"
    return 0
}

# Frames from the clip's own mp4, if they are not there already. Split out of
# fetch_clip so --dir can reach it: a directory of clips assembled elsewhere has
# the videos but not necessarily the frames, and re-deriving them is the same
# step, minus the download that is the only thing a url was ever needed for.
ensure_frames() {   # ensure_frames <name>; returns non-zero on failure
    local name="$1" clip_dir="$DATA_DIR/$1" mp4

    if [ -d "$clip_dir/$FRAMES_SUBDIR" ] && [ -n "$(ls -A "$clip_dir/$FRAMES_SUBDIR" 2>/dev/null)" ]; then
        echo "  frames: have $(ls -1 "$clip_dir/$FRAMES_SUBDIR" | wc -l) frames in $FRAMES_SUBDIR/"
        return 0
    fi

    # <clip>/<clip>.mp4 is the convention every fetched clip follows; the glob is
    # for a directory assembled by hand, where the video may carry its own name.
    # Only when there is exactly one -- picking between two would be a guess.
    mp4="$clip_dir/$name.mp4"
    if [ ! -s "$mp4" ]; then
        local found=()
        while IFS= read -r -d "" f; do found+=("$f"); done \
            < <(find "$clip_dir" -maxdepth 1 -type f -name "*.mp4" -print0 2>/dev/null)
        case "${#found[@]}" in
            1) mp4="${found[0]}" ;;
            0) fail "$name" frames "no frames and no mp4 in $clip_dir"; return 1 ;;
            *) fail "$name" frames "no frames, and ${#found[@]} mp4s in $clip_dir -- \
rename the one to use to $name.mp4"; return 1 ;;
        esac
    fi

    if [ "$DRY" = 1 ]; then
        echo "  frames: would extract $(basename "$mp4") at ${SOURCE_HZ}Hz into $FRAMES_SUBDIR/, fitted to $TARGET_SIZE"
        return 0
    fi
    if ! python "$EXTRACT" --video "$mp4" --save-dir "$clip_dir/$FRAMES_SUBDIR" \
             --hz "$SOURCE_HZ" --size "$TARGET_SIZE" --crop-bias "$CROP_BIAS"; then
        fail "$name" frames "process_ytb.py failed on $mp4"
        return 1
    fi
    [ -n "$(ls -A "$clip_dir/$FRAMES_SUBDIR" 2>/dev/null)" ] || {
        fail "$name" frames "no frames written to $clip_dir/$FRAMES_SUBDIR"; return 1; }
    return 0
}

# --- render -----------------------------------------------------------------
# must_exist=1 (the default) means the clip was named rather than fetched, so a
# missing directory is a typo. At 0 the fetch step is what would have created it,
# and under DRY_RUN it deliberately did not -- reporting that as a failure would
# make a dry run of any genuinely new clip look broken.
submit_clip() {   # submit_clip <name> [must_exist]; returns non-zero on failure
    local VIDEO="$1" must_exist="${2:-1}" clip_dir="$DATA_DIR/$1" out_dir
    out_dir="$clip_dir${RUN:+/$RUN}"

    # Frames are the pipeline's only input; a clip that was downloaded but never
    # run through process_ytb.py would otherwise fail deep inside stage 1.
    if [ ! -d "$clip_dir" ] || [ -z "$(ls -A "$clip_dir/$FRAMES_SUBDIR" 2>/dev/null)" ]; then
        if [ "$must_exist" != 1 ] && [ "$DRY" = 1 ]; then
            echo "  render: would submit VIDEO=$VIDEO DATASET=$DATASET RUN=$RUN (once its frames exist)"
            submitted=$((submitted + 1))
            return 0
        fi
        [ -d "$clip_dir" ] || { fail "$VIDEO" render "no such clip directory $clip_dir"; return 1; }
        fail "$VIDEO" render "no frames in $clip_dir/$FRAMES_SUBDIR"
        return 1
    fi
    # Everything a disabled stage was going to supply from disk. Checked before
    # the done-marker rather than after: a clip that cannot be re-rendered is not
    # "skipped, already done", it is missing an input, and the two want different
    # words. lane_masks is the one whose absence is survivable -- the lifter
    # takes --lane_masks only when the directory is there -- so it warns.
    local missing=()
    while IFS='|' read -r rel label flag; do
        [ -n "$rel" ] || continue
        if [ ! -e "$out_dir/$rel" ]; then
            case "$rel" in
                lane_masks) echo "  render: no $rel/ to reuse; rendering without lanes" ;;
                *) missing+=("$label ($rel, set $flag=1 to make them)") ;;
            esac
        fi
    done < <(reuse_requirements)
    if [ "${#missing[@]}" -gt 0 ]; then
        fail "$VIDEO" render "nothing to reuse in $out_dir: ${missing[*]}"
        return 1
    fi

    local done_dir="$clip_dir${DONE_RUN:+/$DONE_RUN}"
    if [ "${FORCE:-0}" != 1 ] && [ -e "$done_dir/$DONE_MARKER" ]; then
        echo "  render: skip, $done_dir/$DONE_MARKER exists (FORCE=1 to re-run)"
        skipped=$((skipped + 1))
        return 0
    fi

    export VIDEO
    if [ "$DRY" = 1 ]; then
        echo "  render: would submit VIDEO=$VIDEO DATASET=$DATASET RUN=$RUN" \
             "$([ "$NEEDS_GPU" = 1 ] && echo "(gpu)" || echo "(no gpu)")"
    elif [ "${LOCAL:-0}" = 1 ]; then
        echo "  render: running locally"
        if ! bash "$RENDER"; then fail "$VIDEO" render "run_render_vis3d.sh returned non-zero"; return 1; fi
    else
        # --export=ALL hands the job the config above; --job-name feeds the %x in
        # the render script's own --output, so each clip gets its own log file.
        if ! sbatch --job-name="vis3d_${DATASET}_${VIDEO}" --export=ALL \
                    "${SBATCH_EXTRA[@]}" "$RENDER"; then
            fail "$VIDEO" render "sbatch refused the job"
            return 1
        fi
    fi
    submitted=$((submitted + 1))
    return 0
}

# --- what to process --------------------------------------------------------
# Three ways in. Named clips and ALL=1 skip the fetch entirely: they act on what
# is already on disk, so a name that is not there is a typo rather than a clip
# waiting to be downloaded, and it is reported as one.
submitted=0
skipped=0

# Say which stages are actually going to run, once, before fifty clips of
# output. The stage flags are the whole difference between a re-render and a
# re-run and they are set in three places (defaults, RERENDER, environment), so
# printing what won beats reading back the block that set it.
stages=""
[ "$RUN_STAGE1_MASKS" = 1 ]  && stages="$stages masks"
[ "$RUN_STAGE1B_LANES" = 1 ] && stages="$stages lanes"
[ "$RUN_STAGE2_DEPTH" = 1 ]  && stages="$stages depth"
[ "$RUN_STAGE3_LIFT" = 1 ]   && stages="$stages lift+raster"
[ "$RUN_STAGE4_VO" = 1 ]     && stages="$stages vo"
[ "$EXPORT_NAVSIM" = 1 ]     && stages="$stages navsim-export"
echo "stages:${stages:- none}   gpu: $([ "$NEEDS_GPU" = 1 ] && echo yes || echo no)"
[ "$RERENDER" = 1 ] && echo "RERENDER=1: reusing each clip's masks, point maps," \
                            "lane masks and poses; FORCE=${FORCE:-1}"

if [ "$#" -gt 0 ]; then
    echo "== rendering ${#} named clip(s) from $DATA_DIR, into RUN=${RUN:-<clip dir>} =="
    for name in "$@"; do
        echo "-- $name"
        # A named clip that has its video but no frames is a clip waiting for the
        # extract step, not a typo, so derive them rather than refusing. Only
        # under --dir: without it, naming a clip has always meant "this is on
        # disk and ready", and quietly extracting would hide a misspelling.
        if [ -n "$CLIP_DIR" ]; then
            [ -d "$DATA_DIR/$name" ] || { fail "$name" render "no such clip directory $DATA_DIR/$name"; continue; }
            ensure_frames "$name" || continue
            # 0: under DRY_RUN the extract above only said what it would do, so
            # the frames it promised are legitimately not there yet.
            submit_clip "$name" 0
            continue
        fi
        submit_clip "$name"
    done
elif [ "${ALL:-0}" = 1 ]; then
    echo "== rendering every clip in $DATA_DIR, into RUN=${RUN:-<clip dir>} =="
    for d in "$DATA_DIR"/*/; do
        [ -d "$d" ] || continue      # no match: the glob stays literal
        name=$(basename "$d")
        echo "-- $name"
        if [ -n "$CLIP_DIR" ]; then
            ensure_frames "$name" || continue
            submit_clip "$name" 0    # see the named-clip branch above
            continue
        fi
        submit_clip "$name"
    done
else
    [ "${#CLIPS[@]}" -gt 0 ] || { echo "error: the CLIPS array is empty" >&2; exit 1; }
    echo "== processing ${#CLIPS[@]} clip(s) from the CLIPS array =="
    for entry in "${CLIPS[@]}"; do
        # Split on the first comma only: "<name>, <url>". Urls have no commas to
        # speak of, but query strings can, so splitting on the last one or on
        # every one would eventually truncate a link.
        if [[ "$entry" != *,* ]]; then
            fail "<no comma>" input "malformed entry, want \"<name>, <url>\": $entry"
            continue
        fi
        base=$(trim "${entry%%,*}")
        url=$(trim "${entry#*,}")
        if [ -z "$base" ] || [ -z "$url" ]; then
            fail "${base:-<blank>}" input "malformed entry, want \"<name>, <url>\": $entry"
            continue
        fi
        resolve_name "$base" "$url"
        name="$RESOLVED_NAME"
        if [ -z "$name" ]; then
            fail "$base" input "would re-download: $DUPLICATE_OF is already fetched from a different url"
            echo "      on disk:  ${DUPLICATE_URL:-<no source_url recorded>}" >&2
            echo "      in CLIPS: $url" >&2
            echo "      Point the CLIPS entry at the url on disk to resume it, give this" >&2
            echo "      entry its own name, or set ALLOW_DUPLICATE=1 to fetch it alongside" >&2
            echo "      as a separate clip." >&2
            continue
        fi
        [ "$name" = "$base" ] && echo "-- $name" || echo "-- $name  (name $base was taken)"
        fetch_clip "$name" "$url" || continue
        submit_clip "$name" 0
    done
fi

# --- summary ----------------------------------------------------------------
echo
echo "$submitted clip(s) started, $skipped skipped, ${#failures[@]} failed."
if [ "${#failures[@]}" -gt 0 ]; then
    echo "failures:" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
fi
