#!/bin/bash
# Regenerate a clip's 3D boxes end to end:  vis3d.sh <clip|all> [run] [dataset]
#
# The single-clip counterpart to run_render_dataset.sh, minus the download: it
# extracts frames if the clip has none, then submits the same six stages with the
# same --export=ALL convention. Extraction is CPU-only and takes seconds, so it
# runs here; four of the stages want a GPU and this machine has none, so those are
# submitted -- watch my_dump/vis3d_<dataset>_<clip>.out.<jobid>.
#
# Exported rather than just set, for the reason run_render_dataset.sh gives:
# --export=ALL hands the job the *environment*, so an unexported variable never
# reaches it and the render quietly falls back to its own defaults. Every other
# CONFIG name in run_render_vis3d.sh travels the same way:
#
#   vis3d.sh all                         every clip in the dataset, one job each
#   RERENDER=1 vis3d.sh all              re-lift, re-filter and re-render all of
#                                        them from inputs they already hold --
#                                        CPU only, no GPU, no re-detection. This
#                                        is the mode for a change to the lifter,
#                                        the temporal filter, drop_ego_artifacts
#                                        or the renderer.
#   vis3d.sh wrongway                    extract at 10 Hz, then everything
#   RUN_STAGE2_DEPTH=0 vis3d.sh beepbeep keep the point maps already there
#   SOURCE_HZ=2 SUBSAMPLE_STRIDE=1 vis3d.sh beepbeep 3
#                                        the old 2 Hz behaviour, on 2 Hz frames
#   SUBSAMPLE_RUN=2hz SUBSAMPLE_FRAMES=frames_2hz vis3d.sh wrongway
#                                        also deliver a 2 Hz copy of the clip
#   RUN_STAGE1_MASKS=0 RUN_STAGE1B_LANES=0 RUN_STAGE2_DEPTH=0 RUN_STAGE4_VO=0 \
#     RUN_STAGE3_LIFT=1 vis3d.sh wrongway 4
#                                        re-lift, re-filter and re-render from the
#                                        masks, lanes, point maps and poses run 4
#                                        already holds -- CPU only, no re-detection
set -eu

# Absolute, not resolved from $0: sbatch copies a submitted script into the
# node's spool directory, so a sibling path would point at a directory holding
# nothing but that copy.
BASE=/fs/nexus-projects/sim2real/aliu/RAP

# A positional wins over the environment, and the environment over the default,
# so `RUN=3 vis3d.sh beepbeep` means run 3 rather than being silently overwritten
# with the default and sent at an empty directory. `-` not `:-` on RUN, matching
# run_render_vis3d.sh: RUN="" is the clip directory itself, not an unset value.
VIDEO_ARG=${1:?usage: vis3d.sh <clip|all> [run] [dataset]}
export RUN=${2-${RUN-4}}
export DATASET=${3:-${DATASET:-test}}

# --- RERENDER: redraw from what each clip already holds -----------------------
# Stages 1, 1b, 2 and 4 derive a clip's inputs -- masks, lane masks, point maps,
# ego poses -- and none of them reads the lifting, filtering or drawing code. So
# after a change to any of those there is nothing to recompute but stage 3, which
# is cv2/numpy only. On 50 clips that is the difference between minutes on no GPU
# and hours holding 50 cards. Written with :- so each stage is still individually
# overridable: RERENDER=1 RUN_STAGE2_DEPTH=1 re-infers depth and reuses the rest.
#
# Set BEFORE the stage defaults below, which is what lets those defaults see it.
if [ "${RERENDER:-0}" = 1 ]; then
    : "${RUN_STAGE1_MASKS:=0}" "${RUN_STAGE1B_LANES:=0}" "${RUN_STAGE2_DEPTH:=0}"
    : "${RUN_STAGE4_VO:=0}" "${RUN_STAGE3_LIFT:=1}" "${EXPORT_NAVSIM:=0}"
fi

export RUN_STAGE1_MASKS="${RUN_STAGE1_MASKS:-1}"
export RUN_STAGE1B_LANES="${RUN_STAGE1B_LANES:-1}"
export RUN_STAGE2_DEPTH="${RUN_STAGE2_DEPTH:-1}"
export RUN_STAGE3_LIFT="${RUN_STAGE3_LIFT:-1}"
export RUN_STAGE4_VO="${RUN_STAGE4_VO:-1}"
export EXPORT_NAVSIM="${EXPORT_NAVSIM:-1}"

# Stage 3c, the ego/road artifact filter. Exported for the same reason as
# everything above: --export=ALL hands the job the environment, so
# DROP_EGO_ARTIFACTS=0 set but not exported would reach the node as nothing and
# the filter would run anyway, which is the failure mode hardest to notice --
# the boxes come back missing and the setting that was supposed to keep them
# looks like it was ignored.
export DROP_EGO_ARTIFACTS="${DROP_EGO_ARTIFACTS:-1}"
# 0 = the size rule is OFF. It cannot tell a failed fit from a near vehicle in a
# clip whose other detections are distant: at 4.0 it deleted the white Explorer
# directly ahead on four_way (6.01 against a clip median of 1.26) and 114 real
# cars on missing_blinker. Turn it on per clip only after looking at what it takes.
export MAX_CLASS_RATIO="${MAX_CLASS_RATIO:-0}"
export MAX_FRAME_COVER="${MAX_FRAME_COVER:-0.85}"
export COVER_MIN_RATIO="${COVER_MIN_RATIO:-2.0}"
export MAX_FLICKER_TRACK="${MAX_FLICKER_TRACK:-4}"
export MIN_FLICKER_COVER="${MIN_FLICKER_COVER:-0.15}"
# PER-CLIP ONLY. How far above the horizon a bottom-touching box may still reach
# and count as ego bodywork. 0 = entirely below, which is the only globally safe
# value: opposing_crash's face box needs 0.17 and turn_blocker's real Mercedes
# sits at 0.14, so any global setting that clears the first deletes the second
# (779 boxes). Raise it for one clip you have looked at:
#   BONNET_ABOVE_HORIZON=0.20 RERENDER=1 DATASET=CARE_YTB RUN=1 ./vis3d.sh opposing_crash
export BONNET_ABOVE_HORIZON="${BONNET_ABOVE_HORIZON:-0}"
export DROP_BONNET="${DROP_BONNET:-1}"

# Process at 10 Hz, and deliver that. Everything that reasons across time gets
# easier as the frames get closer together and none of it gets harder --
# association gates on how far a box moves between frames, the yaw and extent
# medians get five times the samples over the same stretch of road, and
# UniDepth's metric scale drifts with elapsed time rather than jittering per
# frame. That is a reason to run fast, not a reason to hand back something
# slower; see subsample_clip.py for the pass that used to do the handing back.
#
# SOURCE_HZ and SUBSAMPLE_STRIDE move together: the frame-count knobs in
# run_render_vis3d.sh (MAX_GAP, MAX_FILL, the median windows, and
# inversely the two rate limits) are all scaled by SUBSAMPLE_STRIDE because their
# intent is a duration, so raising SOURCE_HZ without it would leave every temporal
# window meaning a fifth of the time it was tuned for.
#
# Delivering a second, lower-rate copy is a separate question, and the answer here
# is no. Stage 5 used to subsample the finished run into <clip>/2hz, its frames
# into <clip>/frames_2hz, and then link <clip>/1 and <clip>/frames at the pair --
# four names for two things, and a 2 Hz clip nothing downstream was reading. An
# empty SUBSAMPLE_RUN turns that stage off, and the clip directory is then one
# frames/ and one run, each under its own name. The navsim export at the bottom
# of run_render_vis3d.sh writes from the processed run instead, at SOURCE_HZ.
#
# That is also why FRAMES_SUBDIR is plain frames/ rather than frames_10hz/: the
# rate-tagged names existed so a clip could hold two rates at once without one
# overwriting the other, and a clip only ever holds one now. run_render_dataset.sh
# keeps the tagged defaults, since a batch run may still want both.
# RUN_STAGE1_MASKS=0 RUN_STAGE1B_LANES=0 RUN_STAGE2_DEPTH=0 RUN_STAGE4_VO=0   RUN_STAGE3_LIFT=1 ./vis3d.sh wrongway 4
export SOURCE_HZ="${SOURCE_HZ:-10}"
export SUBSAMPLE_STRIDE="${SUBSAMPLE_STRIDE:-5}"
export FRAMES_SUBDIR="${FRAMES_SUBDIR:-frames}"
export SUBSAMPLE_FRAMES="${SUBSAMPLE_FRAMES-}"
export SUBSAMPLE_RUN="${SUBSAMPLE_RUN-}"

# One clip: extract frames if it has none, then submit it. Extraction is CPU and
# takes seconds, so it happens here rather than burning a GPU allocation on it.
submit_clip() {
    export VIDEO="$1"
    CLIP="$BASE/data/$DATASET/$VIDEO"
    OUT="$CLIP${RUN:+/$RUN}"

    if [ -n "$(ls -A "$CLIP/$FRAMES_SUBDIR" 2>/dev/null)" ]; then
        echo "frames: have $(ls -1 "$CLIP/$FRAMES_SUBDIR" | wc -l) in $FRAMES_SUBDIR/"
    else
        [ -s "$CLIP/$VIDEO.mp4" ] || {
            echo "error: no frames and no $CLIP/$VIDEO.mp4" >&2; return 1; }
        python "$BASE/vis3d/process_ytb.py" --video "$CLIP/$VIDEO.mp4" \
            --save-dir "$CLIP/$FRAMES_SUBDIR" --hz "$SOURCE_HZ" \
            --size "${TARGET_SIZE:-1920x1080}" --crop-bias "${CROP_BIAS:-0.5}"
    fi

    # A re-render reuses inputs rather than deriving them, so a clip that never
    # got them cannot be re-rendered. Skipped with a line rather than submitted
    # to fail on the node, because across 50 clips the difference is one visible
    # message against fifty logs nobody reads.
    if [ "${RERENDER:-0}" = 1 ] && [ ! -e "$OUT/mask_results_preds.json" ]; then
        echo "skip $VIDEO: no mask_results_preds.json in $OUT -- never had stage 1"
        return 0
    fi

    sbatch --job-name="vis3d_${DATASET}_${VIDEO}" --export=ALL \
        "$BASE/scripts/vis3d/run_render_vis3d.sh"
}

if [ "$VIDEO_ARG" = all ]; then
    # One job per clip, not one job over all of them: a clip that fails takes
    # only itself down, and 50 short CPU jobs schedule far better than one long
    # one. Directory order is the dataset's own, so a re-run submits the same
    # list in the same order.
    CLIPS=$(cd "$BASE/data/$DATASET" && ls -d */ 2>/dev/null | sed 's:/$::')
    [ -n "$CLIPS" ] || { echo "error: no clips in $BASE/data/$DATASET" >&2; exit 1; }
    echo "submitting $(echo "$CLIPS" | wc -w) clip(s) from $DATASET, run '${RUN}'"
    [ "${RERENDER:-0}" = 1 ] && echo "RERENDER: stage 3 only (lift, smooth, filter, raster)"
    failed=0
    for clip in $CLIPS; do
        echo "== $clip"
        submit_clip "$clip" || failed=$((failed + 1))
    done
    [ "$failed" = 0 ] || echo "$failed clip(s) could not be submitted" >&2
    exit 0
fi

submit_clip "$VIDEO_ARG"
