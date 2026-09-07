#!/bin/bash

#SBATCH --job-name=vis3d_pipeline
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Covers render *and* ego motion now that they are one job.
#SBATCH --time=07:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma

############################### CONFIG ###############################
VIDEO="${VIDEO:-beepbeep}"       # clip under data/$DATASET/
DATASET="${DATASET:-test}"        # dataset dir under data/ (YTB, CARE_YTB, ...)
RUN="${RUN-3}"                   # output subdir ("" = clip dir itself)

# Stages to run. 1 = GPU needed, so sbatch this rather than running it here.
RUN_STAGE1_MASKS=${RUN_STAGE1_MASKS:-0}   # objects (GroundingDINO+SAM)
RUN_STAGE1B_LANES=${RUN_STAGE1B_LANES:-0} # lane dividers (YOLOPv2)
# 1 = fold hand-drawn boxes into stage 1's output when the clip has any. Costs
# nothing on a clip with no manual_boxes.json, which is why it is on by default:
# a clip that has been corrected once stays corrected across every later
# re-render without anyone having to remember a flag. See vis3d/annotate_boxes.py.
APPLY_MANUAL_BOXES=${APPLY_MANUAL_BOXES:-1}
MANUAL_BOX_IOU=${MANUAL_BOX_IOU:-0.5}     # detector box overlapping a manual one by this much is dropped
# On the frames a drawn track does NOT cover, a detection overlapping where the
# track just was is given that track's id, so a detector that caught the object
# only in glimpses has those glimpses carried by the annotation instead of
# deleted by MIN_TRACK_LEN. ADOPT_GAP=0 turns it off.
MANUAL_ADOPT_IOU=${MANUAL_ADOPT_IOU:-0.3} # overlap with the track's last box that claims a detection
MANUAL_ADOPT_GAP=${MANUAL_ADOPT_GAP:-5}   # frames a track may go unseen before its last box is too stale
RUN_STAGE2_DEPTH=${RUN_STAGE2_DEPTH:-0}   # point maps (UniDepth)
RUN_STAGE3_LIFT=${RUN_STAGE3_LIFT:-0}     # lift boxes + rasterize overlays (CPU)
RUN_STAGE4_VO=${RUN_STAGE4_VO:-1}         # ego trajectory (visual odometry)
# Verify that every output which draws over a frame still agrees with that
# frame about which channel is red. 0 skips the check; it does not change
# what gets rendered either way. See check_channel_order.
CHECK_COLOR=${CHECK_COLOR:-1}
EXPORT_NAVSIM=${EXPORT_NAVSIM:-1}         # navsim-format log; needs stage 3 and stage 4 output

# Where each skipped stage borrows its output from: a run subdir, or "" for
# the clip dir. They need not be the same place. Leaving one "" when the
# artifact is already in this run's own directory is fine -- see link_reused.
REUSE_MASKS_FROM="${REUSE_MASKS_FROM-}"
REUSE_LANES_FROM="${REUSE_LANES_FROM-}"              # "" + stage off = render without lanes
REUSE_DEPTH_FROM="${REUSE_DEPTH_FROM-}"

# 1 = temporally filter each track's heading and extent after lifting. Boxes are fit one
# frame at a time, so a heading can jump tens of degrees and come straight back
# (15% of frames moved >10 deg, 2% >30 deg), which the per-face palette turns
# into visible colour flicker as the top face grows or vanishes. Extent is worse
# still -- 12% of frames change a box's size by over a quarter and the 99th
# percentile is a doubling, which reads as the box wobbling like jello. Both
# filters are medians, so they delete those spikes outright while a genuine turn
# or collision passes through with no lag at all -- and then a rate limit on top
# of each, for the errors that never come back. See MAX_YAW_RATE / MAX_SIZE_RATE.
#
# Centre is never touched: lift_frames_to_3d.py anchors it to the mask, and
# moving it would slide the box off its own detection. Extent shrinkage is capped
# by MAX_SHRINK for the same reason -- boxes are lifted already shrunk to the
# edge of their coverage floor, so shrinking is the only direction that loses it.
# Process at a high rate, deliver at a low one. 1 = off (process and deliver at
# the same rate, the historical behaviour). 5 = the frames are 10 Hz and the clip
# is delivered at 2 Hz.
#
# Everything here that reasons across time gets easier as the frames get closer
# together and none of it gets harder: association gates on how far a box moves
# between frames, the yaw and extent medians get five times the samples over the
# same stretch of road, and UniDepth's metric scale -- which drifts with elapsed
# time rather than jittering per frame -- drifts a fifth as far per link. The
# cost is five times the stage-1 and stage-2 compute.
#
# The two halves of this are separate knobs, and conflating them is the trap.
# SUBSAMPLE_STRIDE is the ratio between the rate the pipeline RUNS at and the
# rate its frame-count windows below were tuned at, so it scales MAX_GAP,
# MAX_FILL and the two medians whether or not anything is
# delivered at the lower rate -- turning it down to 1 to stop the delivery would
# quietly retune every temporal filter to a fifth of the duration it wants.
# SUBSAMPLE_RUN is where the lower-rate copy goes, and empty means nowhere: no
# second run directory, no second frames directory, and none of the frames/ and
# 1/ links stage 5 used to point at them.
SUBSAMPLE_STRIDE=${SUBSAMPLE_STRIDE:-1}
SUBSAMPLE_FRAMES=${SUBSAMPLE_FRAMES-frames_2hz}   # subdir for the delivered frames
SUBSAMPLE_RUN=${SUBSAMPLE_RUN-2hz}                # run subdir, "" = deliver no second copy

SMOOTH_TRACKS=${SMOOTH_TRACKS:-1}
YAW_WINDOW=${YAW_WINDOW:-$(( (5 * SUBSAMPLE_STRIDE) | 1 ))}                     # frames in the heading median window (odd)
YAW_INLIER_DEG=15                # headings within this of the median are averaged
SIZE_WINDOW=${SIZE_WINDOW:-$(( (5 * SUBSAMPLE_STRIDE) | 1 ))}                    # frames in the extent median window (odd)
SIZE_INLIER_FRAC=0.25            # extents within this fraction of the median are averaged
MAX_SHRINK=0.90                  # extent filter may not shrink a box below this x its lifted size
# The medians above remove what comes back; these remove what does not. The
# lifter picks a heading from 36 candidates whose IoU spread is barely above the
# roughness of the mask, so a bad frame can hand the argmax to a rival 20-45 deg
# away and every frame after it then agrees -- a step no median can see, because
# two frames later the wrong heading owns the window. Rate-limiting the
# derivative does see it: on wrongway/4 every track's heading is steady to
# 0.7 deg/frame at p90 once the ego's own rotation is out of it, while every flip
# moves at least 11 deg in one frame. Same for extent, where the gap is p98 29%
# against a next value of 44%. See smooth_boxes.MAX_YAW_RATE_DEG / MAX_SIZE_RATE.
#
# Both are per-frame allowances, so both scale DOWN as the frames get denser --
# the inverse of the windows above, which are durations and scale up. The
# constants are quoted at the 2 Hz baseline the rest of this block is tuned at:
# 40 deg and a 4.5x size ratio per frame there, which is 8 deg and 1.35x at the
# 10 Hz the pipeline actually runs at. At 2 Hz the size limit is wide enough to
# be no limit, which is the honest answer -- half a second is long enough that a
# real approach and a blown-up mask are not distinguishable by rate alone.
MAX_YAW_RATE=${MAX_YAW_RATE:-$(( 40 / SUBSAMPLE_STRIDE ))}
MAX_SIZE_RATE=${MAX_SIZE_RATE:-$(awk -v s="$SUBSAMPLE_STRIDE" \
    'BEGIN { printf "%.3f", 1.35 ^ (5 / s) - 1 }')}
# MAX_PX is now a *ceiling* on the association gate rather than the gate itself:
# smooth_boxes gates on GATE_EXTENT_FRAC of each object's own projected size, and
# clamps that to [MIN_PX, MAX_PX]. A single absolute gate cannot serve a frame
# holding both a truck 12 m away and a hatchback at 40 m -- on beepbeep the truck's
# centre moved 127 px between two frames, 17% of its own width and obviously the
# same object, while for the 40 px car in the same frame 127 px is three times its
# whole extent. At a flat 120 px the truck's track broke exactly there, a track of
# distant cars adopted its box, and the orphan interpolated a second 7.5 m box
# across the hole it thought it had: two boxes on one truck.
#
# MAX_SIZE_RATIO is what makes the wider gate safe -- two boxes whose silhouettes
# differ by more than it are not the same object however close their centres are
# (the truck was 464 px against the 48 px car that took it, 25 px away) -- and
# SIZE_WEIGHT ranks the admissible candidates so the right-sized one wins over the
# merely nearer one. Measured on beepbeep against mask-IoU identity: adjacent-frame
# links joining unrelated objects fell from 23/193 to 3/177, and links that should
# have been made and were not from 13 to 7. Both error types improve, so the extra
# track fragmentation is not a trade -- the flat gate was keeping boxes by gluing
# them to the wrong object.
#
# MAX_GAP is how many frames a track may go undetected before it is closed.
# Skip detections below this score at lift time. 0 = lift everything the
# detector emitted, which is the historical behaviour and stays the default:
# GroundingDINO's floor is already 0.30, and on a busy clip the 0.30-0.45 band is
# mostly real distant traffic. It is background clutter on a clip with an
# obstructed verge -- buick_nearmiss frame 21 lifts six boxes at 0.31-0.40 out of
# a hedge, as 0.85-1.2 m "cars" facing backwards -- so this is a per-clip knob.
# MIN_TRACK_LEN used to hide these by deleting their short tracks; it no longer
# does, which is the price of it no longer deleting real ones.
MIN_BOX_SCORE=${MIN_BOX_SCORE:-0}
# Scaled by SUBSAMPLE_STRIDE, because every one of these is a frame COUNT whose
# intent is a duration. At 2 Hz "4 frames" is two seconds; at 10 Hz it is four
# tenths of one, and leaving them alone would quietly change what the filter
# means rather than what rate it runs at. Overriding any of them in the
# environment wins, and then it is your number, unscaled.
# NOT scaled, for the reason the MAX_PX paragraph above gives: this stopped being
# the gate when GATE_EXTENT_FRAC took over, and a ceiling is not a per-frame
# motion allowance. 250 was the old absolute gate at 2 Hz; scaling it to 50 and
# flooring at 100 left a ceiling *below* the gate it is supposed to bound, so for
# any object over 200 px across the extent-relative gate was thrown away and the
# flat cap this whole paragraph argues against was back.
#
# It bites exactly where the extent gate was introduced to help. On the ambulance
# clip the ambulance crosses the intersection at 405 -> 506 px, wants a gate of
# 0.5 * 506 = 253 px, gets 100, and moves 103 px between frames 72 and 73 -- so
# its track breaks by 3 px at the closest, largest, most important moment. The
# two halves are then filtered as different objects: the 25-frame extent median
# on the first half still has the distant frames in its window and holds the box
# at 2.9 m and the wrong heading, while the second half starts fresh at 4.9 m and
# the true one, which is the box jumping size and spinning between 72 and 73.
# Whole and ungated, the same track runs 3.7 -> 5.0 m with the heading steady.
MAX_PX=${MAX_PX:-250}
MAX_GAP=${MAX_GAP:-$(( 4 * SUBSAMPLE_STRIDE ))}
# Not scaled by SUBSAMPLE_STRIDE: GATE_EXTENT_FRAC is a fraction of the object,
# and MAX_SIZE_RATIO a ratio between two of them, so neither is a frame count
# whose meaning changes with the rate. Processing faster only moves each pair
# further inside the same gate.
GATE_EXTENT_FRAC=${GATE_EXTENT_FRAC:-0.5}   # gate as a fraction of the object's projected size
MIN_PX=${MIN_PX:-40}                        # floor under that gate, in pixels
MAX_SIZE_RATIO=${MAX_SIZE_RATIO:-2.5}       # sizes further apart than this are different objects
SIZE_WEIGHT=${SIZE_WEIGHT:-1.0}             # size mismatch in the cost, octaves per gate width
# Fraction of its silhouette a box needs inside the frame before its class label
# joins its track's vote. Apparent size says how readable a detection is, but only
# while the object still fits: the beepbeep truck reads as truck, truck, truck, bus
# at 12-16 m, then "car" in all ten frames from 21 on, where it is the largest it
# ever is and only 64% -> 45% -> 18% of it is in shot. A featureless slab of cab
# door is what a car looks like too, so those frames are outvoted rather than
# trusted. Whole clip: 0 non-car labels before, 18 truck boxes after.
VOTE_MIN_VISIBLE=${VOTE_MIN_VISIBLE:-0.9}
# Drop every box of a track seen in fewer than this many frames. A detection
# that fires once and is gone -- typically oncoming traffic across the divider,
# caught for a single frame -- pops a box into existence and out again, which
# reads worse in a video than simply not drawing it. 0 keeps them all.
#
# 2 rather than 3: this deletes real detections, and it is only worth that when
# association is trustworthy enough for a short track to really mean a flicker.
# With the gates above widened, 2 costs nothing on this clip (0 boxes dropped)
# and still catches the single-frame case it was written for.
#
# NOT scaled by SUBSAMPLE_STRIDE, unlike MAX_GAP and MAX_FILL below, and this is
# the one place in this block where a frame count is not standing in for a
# duration. What it counts is how many times the detector fired, and "fired once
# and is gone" is one frame at any rate. Scaled to 10 it stopped being a flicker
# filter and became a one-second minimum lifetime, which no object owes: it
# deletes whatever the ego drives *past*, because a car at the kerb 5 m away
# sweeps out of a 50-degree frame in well under a second while the identical car
# 40 m ahead sits in it for six. On the ambulance clip that removed 181 of 1389
# boxes across 96 of 155 frames, and frame 0 lost every parked car on the right
# -- four tracks of 7 to 9 detections, best scores 0.61 and 0.60 -- while the
# left-hand row, further off and slower to leave, survived at 17 to 65. A filter
# whose effect is "near objects are dropped and far ones kept" is measuring
# range, not confidence. At 3 the same clip drops 10 boxes instead of 181.
MIN_TRACK_LEN=${MIN_TRACK_LEN:-3}
# ... and the mirror image: interpolate a track's box across dropouts of up to
# this many consecutive frames, so an object the detector momentarily loses does
# not blink out and back. 0 leaves the holes.
#
# 1 rather than 3, because a filled box is invented rather than measured and this
# footage cannot afford many: at 3 the same clip fabricated 37 boxes against 245
# lifted, and the floating box in mid-air on frame 17 was one of them. At 1 it
# fabricates 9, which covers the one-frame dropout the setting is for without
# carrying a track across a gap long enough for the object to have gone.
MAX_FILL=${MAX_FILL:-$(( 1 * SUBSAMPLE_STRIDE ))}

# --- Stage 3c: drop the boxes the lifter fits to the ego and the road ---------
# Three failure modes put a box in the ego's own face, and none of them is a
# detection of anything: a mask that bled onto the tarmac and came back as one
# enormous slab; the ego's own bonnet read as a vehicle ahead; a box so badly
# ranged that its projection encloses the entire frame. On yield_runway the last
# of those accounted for 72 of 190 boxes and tinted every overlay.
#
# It runs here, after the temporal filter and before the raster, because
# smooth_boxes votes a track's settled class and extent -- and it is the settled
# extent this compares against the clip's own median. Filtering first would
# measure boxes the smoother is about to resize.
#
# It also has to run inside stage 3 rather than as a pass over finished clips:
# stage 3 rewrites boxes_3d.json from the lift every time it runs, so a filter
# applied by hand afterwards survives exactly until the next re-render and then
# vanishes without saying so -- the same trap manual_boxes_3d.json exists to
# avoid. See vis3d/drop_ego_artifacts.py for what each threshold was measured on.
DROP_EGO_ARTIFACTS=${DROP_EGO_ARTIFACTS:-1}
# 0 = off. A clip's median length conflates near and far -- four_way's cars are
# 1.26 across the clip and the Explorer 18 m ahead is 6.01 -- so any threshold
# that catches a failed fit also catches the nearest real vehicle. Left in place
# because it is occasionally the right tool on one clip, but never by default.
MAX_CLASS_RATIO=${MAX_CLASS_RATIO:-0}
MAX_FRAME_COVER=${MAX_FRAME_COVER:-0.85}   # coverage rule: fraction of the frame
COVER_MIN_RATIO=${COVER_MIN_RATIO:-2.0}    # ...and the size gate inside it
# A big box whose whole track is this short found something for a moment and lost
# it. Catches the bike_crosswalk slab, which is geometrically indistinguishable
# from a real SUV and lives for 3 frames of 358. 0 = off.
MAX_FLICKER_TRACK=${MAX_FLICKER_TRACK:-4}
MIN_FLICKER_COVER=${MIN_FLICKER_COVER:-0.15}
# Per-clip only -- see vis3d.sh. 0 is the only globally safe value.
BONNET_ABOVE_HORIZON=${BONNET_ABOVE_HORIZON:-0}
DROP_BONNET=${DROP_BONNET:-1}              # 0 = size/coverage rules only

LANE_THRESHOLD=0.5               # lane-line probability cut
OVERLAY_ALPHA=0.6                # raster opacity over the frame

# 1 = annotate vis3d_overlay/ with per-box diagnostics: amber sight lines from
# the ego origin out to each object's ground track, a riser up to the box
# centre, and the fitted heading as an arrow -- red when that heading has
# collapsed onto the sight line (the box is then a rod pointed at the camera,
# fitted to depth smear rather than to the object), green otherwise. Labels
# carry the fitted extent in metres, so a box that merely looks small on screen
# can be told apart from one that is small in metres. Stage 3 also prints a
# size sanity check against per-class priors, which is what catches a wrong
# focal length -- the one error nothing else in the pipeline can see.
DEBUG_OVERLAY=0

# --- Stage 4 and the export ---
VO_METHOD="${VO_METHOD:-openvo}" # pointcloud (CPU, metric) | openvo (GPU, learned)
KEEP_DEPTH=${KEEP_DEPTH:-1}      # 1 = keep point maps after VO (~25 MB/frame)
# Empty = write into the run directory itself, so the log and its sensor blobs
# land next to the boxes_3d.json and ego_poses.txt they were built from. Set an
# absolute path to write into a shared tree instead. Either way the log ends up
# at <root>/navsim_logs/$SPLIT/<clip>.pkl; scripts/data/generate_ras_logs.py is
# what links the per-clip copies into one directory for SceneLoader.
DATASET_ROOT="${DATASET_ROOT-}"
SPLIT="${SPLIT:-video}"
SOURCE_HZ=${SOURCE_HZ:-2}        # rate the frames were extracted at (process_ytb.py --hz)

# Stage 1 masks cache their vocabulary: masks from before 'traffic light'
# joined infer_frames.py contain none, so reusing them renders none.
# Stage 1b needs weights/yolopv2.pt:
#   curl -L -o weights/yolopv2.pt \
#     https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
######################################################################

set -euo pipefail
eval "$(conda shell.bash hook)"

# One env for every stage. `set +u` around the activation because conda's own
# activation scripts read unset variables (binutils' ADDR2LINE and friends),
# which is fatal under `set -u`.
set +u
conda activate vis3d
set -u

BASE=/fs/nexus-projects/sim2real/aliu/RAP
PROJECT_ROOT=$BASE/vis3d
cd $PROJECT_ROOT

VIDEO_DIR=$BASE/data/$DATASET/$VIDEO
# Shared input, not per-run. FRAMES_SUBDIR rather than a full FRAMES_DIR so one
# setting works across a batch: the directory differs per clip, the subdirectory
# name does not. Set it to run a clip through the pipeline at a different
# extraction rate (frames_10hz) without disturbing the frames/ the existing run
# was built from.
FRAMES_DIR=$VIDEO_DIR/${FRAMES_SUBDIR:-frames}
OUTPUT_DIR=$VIDEO_DIR${RUN:+/$RUN}

# Frames are the only input every stage reads, and an empty frames/ nearly always
# means DATASET or VIDEO is not what the caller believed -- a wrapper that set one
# of them without exporting it, or a clip filed under a different dataset. Checked
# before the mkdir so a wrong DATASET is named here, in one line, instead of
# stranding an empty output tree under it and dying inside stage 1 a GPU-minute
# later with a bare FileNotFoundError.
if [ -z "$(ls -A "$FRAMES_DIR" 2>/dev/null)" ]; then
    echo "error: no frames at $FRAMES_DIR" >&2
    echo "       DATASET=$DATASET  VIDEO=$VIDEO  RUN=${RUN:-<clip dir>}" >&2
    elsewhere=$(find "$BASE/data" -mindepth 2 -maxdepth 2 -type d -name "$VIDEO" \
                     -not -path "$VIDEO_DIR" 2>/dev/null)
    if [ -n "$elsewhere" ]; then
        echo "       a clip of this name does exist under another dataset:" >&2
        printf '         %s\n' $elsewhere >&2
        echo "       set DATASET to that one, and export it if you set it in a wrapper." >&2
    fi
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# $1: run subdir to borrow from, $2: filename.
# Writes follow symlinks -- never enable a stage whose output is linked in.
link_reused () {
    local reuse_dir="$VIDEO_DIR${1:+/$1}"
    if [ "$OUTPUT_DIR" = "$reuse_dir" ]; then return; fi
    # Nothing named to borrow from, but the artifact is already sitting in this
    # run's own directory -- the normal case once the render half has been run
    # and only the ego-motion half is being re-run. Leave it where it is rather
    # than hunting for a copy in the clip dir that was never meant to exist.
    if [ -z "$1" ] && [ -e "$OUTPUT_DIR/$2" ]; then return; fi
    if [ ! -e "$reuse_dir/$2" ]; then
        if [ -L "$reuse_dir/$2" ]; then   # dangling link: -e is false for it too
            echo "Cannot reuse $2: $reuse_dir/$2 -> $(readlink "$reuse_dir/$2") is missing." >&2
        else
            echo "Cannot reuse $2: $reuse_dir/$2 does not exist." >&2
        fi
        exit 1
    fi
    ln -sfn "$(realpath --relative-to="$OUTPUT_DIR" "$reuse_dir/$2")" "$OUTPUT_DIR/$2"
}

# Colour-order guard.
#
# Every stage that draws on a frame writes the photograph back out underneath
# its overlay, so the result must still agree with the source frame about which
# channel is red. That agreement breaks the moment a BGR array from cv2 is
# handed to PIL, or the reverse -- and it breaks silently: the image is well
# formed, the right size, and merely the wrong colour. The way this was actually
# caught was someone noticing a sunset that should not have been one.
#
# Checked against the source frame rather than against any notion of what a
# scene ought to look like. A genuinely orange sky and a swapped blue one are
# indistinguishable in isolation -- two CARE clips tripped exactly that false
# positive -- but neither can disagree with the frame it was drawn from.
check_channel_order() {   # check_channel_order <output-dir> <frames-dir> <label>
    [ "$CHECK_COLOR" = 1 ] || return 0
    [ -d "$1" ] || return 0
    python - "$1" "$2" "$3" <<'COLORCHECK'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out_dir, frames_dir, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
suffixes = (".jpg", ".jpeg", ".png")
names = [p.name for p in sorted(out_dir.iterdir()) if p.suffix.lower() in suffixes]
names = [n for n in names if (frames_dir / n).exists()]
if not names:
    print(f"  colour check {label}: nothing to compare against {frames_dir}")
    raise SystemExit(0)

# Spread over the clip rather than taking the first few. A swap is global to an
# image, so a handful of frames settles it -- but one vis/ really did mix both
# orders frame to frame, depending on whether the detector had found anything,
# and sampling only the front of the clip would have called that one clean.
step = max(1, len(names) // 8)
swapped, kept = [], []
for name in names[::step]:
    a = np.array(Image.open(out_dir / name).convert("RGB")).astype(float)
    b = np.array(Image.open(frames_dir / name).convert("RGB")).astype(float)
    if a.shape != b.shape:
        continue
    # Per-pixel medians, not whole-frame means. The overlay is the whole reason
    # this file differs from its source, and it is drawn in saturated blues and
    # magentas: on a frame where masks cover 15% of the pixels they pull the mean
    # red down about 10 levels and the mean blue up about 13, which is enough to
    # flip a comparison of means and condemn a perfectly correct image. Measured
    # on blocker/000190.jpg, where outside the masks the two agree to 0.6 of a
    # level. A swap is global, so the median pixel carries it; an overlay covers
    # a minority, so the median pixel ignores it.
    step = max(1, a.shape[0] // 200)
    a, b = a[::step, ::step], b[::step, ::step]
    # The 10th percentile of the per-pixel difference, not the median. Only the
    # pixels the overlay did not touch can say anything about channel order, and
    # they are by construction the closest-matching ones -- so a low percentile
    # reads them whatever fraction of the frame the masks cover. The median is
    # not low enough: snow_crash/000224.jpg has two false detections covering 78%
    # of the frame, which puts the median pixel inside the overlay and made a
    # correct image look swapped.
    def gap(x, y):
        return np.percentile(np.abs(x - y), 10)
    same = gap(a[..., 0], b[..., 0]) + gap(a[..., 2], b[..., 2])
    swap = gap(a[..., 0], b[..., 2]) + gap(a[..., 2], b[..., 0])
    # A margin as well as an ordering: on a grey frame the two are both near zero
    # and their order means nothing.
    if swap + 4.0 < same:
        swapped.append(name)
    else:
        kept.append(name)

if swapped:
    total = len(swapped) + len(kept)
    print(f"  colour check {label}: FAILED -- red and blue are transposed against "
          f"{frames_dir} in {len(swapped)} of {total} sampled frame(s)")
    print(f"    first: {swapped[0]}")
    print("    A cv2 array (BGR) reached PIL, or a PIL array (RGB) reached cv2.imwrite.")
    print("    Convert at that boundary; do not compensate by swapping a palette.")
    raise SystemExit(1)
print(f"  colour check {label}: ok ({len(kept)} frame(s) sampled)")
COLORCHECK
}

# --- Stage 1: object masks ---
# GroundingDINO's custom CUDA op is deliberately *not* built: its source includes
# <THC/THCAtomics.cuh>, a header PyTorch deleted after 1.12, so the extension
# cannot compile against any modern torch -- which is what used to pin this
# stage to a separate py3.8/torch-1.12 env ('zs3d'). ms_deform_attn.py now
# falls back to GroundingDINO's own pure-PyTorch deformable attention, which
# still runs on the GPU and reproduced the compiled op's boxes and scores to
# six decimals on the same frame. No compiler, no cuda module, no build step.
if [ "$RUN_STAGE1_MASKS" = 1 ]; then
    python infer_frames.py --frames_dir $FRAMES_DIR --output_dir $OUTPUT_DIR --save_vis
    # Checked here, before stages 2-4 spend a GPU on top of it.
    check_channel_order "$OUTPUT_DIR/vis" "$FRAMES_DIR" "vis/" || exit 1
else
    link_reused "$REUSE_MASKS_FROM" mask_results_preds.json
fi

# --- Stage 1c: hand-drawn boxes ---
# Merged into stage 1's output rather than applied later, so a manual box is
# lifted, smoothed and exported by exactly the same code as a detected one.
# Safe after a link_reused: the merge writes its result over the *link*, leaving
# the run it was borrowed from with its own untouched copy.
if [ "$APPLY_MANUAL_BOXES" = 1 ] && [ -e "$OUTPUT_DIR/manual_boxes.json" ]; then
    python apply_manual_boxes.py --output_dir "$OUTPUT_DIR" --frames_dir "$FRAMES_DIR" \
        --iou "$MANUAL_BOX_IOU" \
        --adopt_iou "$MANUAL_ADOPT_IOU" --adopt_gap "$MANUAL_ADOPT_GAP"
fi

# --- Stage 1b: lane dividers ---
# Needs nothing but cv2/numpy/torch to load the TorchScript checkpoint; it
# lived in a separate env only because the old stage-1 pin held this one at
# torch 1.12, which cannot load it.
if [ "$RUN_STAGE1B_LANES" = 1 ]; then
    python detect_lanes.py --frames_dir $FRAMES_DIR --output_dir $OUTPUT_DIR \
        --threshold $LANE_THRESHOLD
elif [ -n "$REUSE_LANES_FROM" ]; then
    link_reused "$REUSE_LANES_FROM" lane_masks
fi
# Lanes are optional: lift them only if masks actually landed.
if [ -e "$OUTPUT_DIR/lane_masks" ]; then LANE_FLAG="--lane_masks"; else LANE_FLAG=""; fi

# --- Stage 2: point maps ---
# The env's torch 2.2.0 / numpy 1.26.4 / xformers 0.0.24 pins exist for this
# stage -- see requirements.txt. UniDepth's decoder uses
# xformers' NystromAttention, which xformers deleted along with the rest of its
# research components after 0.0.24, so the pin is what makes this stage share
# an env at all rather than a preference for old torch.
if [ "$RUN_STAGE2_DEPTH" = 1 ]; then
    python infer_unidepth_on_frames.py --frames_dir $FRAMES_DIR --output_dir $OUTPUT_DIR
else
    link_reused "$REUSE_DEPTH_FROM" samples-pseudodepth
fi

# --- Stage 3: lift + rasterize (CPU) ---
# Uses only cv2/numpy/tqdm/pycocotools; needs no GPU even though the env
# carries torch for the stages above.
if [ "$RUN_STAGE3_LIFT" = 1 ]; then
    # --frames_dir: traffic-light states are read off the frame pixels, and it
    # drives the lane pass's frame list.
    python lift_frames_to_3d.py --output_dir $OUTPUT_DIR --frames_dir $FRAMES_DIR \
        --min_score $MIN_BOX_SCORE $LANE_FLAG

    # --- Stage 3b: temporal heading + extent filter (CPU) ---
    # Rewrites boxes_3d.json in place: --in_place leaves every surviving box where the
    # lifter anchored it and preserves every other field, so this is safe to re-run
    # without re-lifting. Still idempotent with MIN_TRACK_LEN and MAX_FILL on -- a
    # second pass re-associates the survivors into the same tracks, all of them long
    # enough and none of them with a hole left to fill.
    if [ "$SMOOTH_TRACKS" = 1 ]; then
        python smooth_boxes.py --output_dir $OUTPUT_DIR --in_place \
            --max_px $MAX_PX --max_gap $MAX_GAP \
            --gate_extent_frac $GATE_EXTENT_FRAC --min_px $MIN_PX \
            --max_size_ratio $MAX_SIZE_RATIO --size_weight $SIZE_WEIGHT \
            --vote_min_visible $VOTE_MIN_VISIBLE \
            --yaw_window $YAW_WINDOW --yaw_inlier_deg $YAW_INLIER_DEG \
            --size_window $SIZE_WINDOW --size_inlier_frac $SIZE_INLIER_FRAC \
            --max_shrink $MAX_SHRINK --min_track_len $MIN_TRACK_LEN --max_fill $MAX_FILL \
            --max_yaw_rate $MAX_YAW_RATE --max_size_rate $MAX_SIZE_RATE \
            --out $OUTPUT_DIR/boxes_3d.json
    fi

    # --- Stage 3c: drop ego/road artifacts (CPU) ---
    # In place, keeping the unfiltered boxes as boxes_3d.prefilter.json, so what
    # this removed can always be looked at without re-lifting.
    if [ "$DROP_EGO_ARTIFACTS" = 1 ]; then
        if [ "$DROP_BONNET" = 1 ]; then BONNET_FLAG=""; else BONNET_FLAG="--no_bonnet"; fi
        python drop_ego_artifacts.py --output_dir $OUTPUT_DIR \
            --max_class_ratio $MAX_CLASS_RATIO --max_frame_cover $MAX_FRAME_COVER \
            --cover_min_ratio $COVER_MIN_RATIO \
            --max_flicker_track $MAX_FLICKER_TRACK \
            --min_flicker_cover $MIN_FLICKER_COVER \
            --bonnet_above_horizon $BONNET_ABOVE_HORIZON \
            $BONNET_FLAG
    fi

    # raster_frames.py imports renderer.py locally, so run it from visualization/.
    if [ "$DEBUG_OVERLAY" = 1 ]; then DEBUG_FLAG="--debug"; else DEBUG_FLAG=""; fi
    (cd visualization && python raster_frames.py --output_dir $OUTPUT_DIR \
        --frames_dir $FRAMES_DIR --overlay_alpha $OVERLAY_ALPHA $DEBUG_FLAG)
    # vis3d_overlay is the render blended over the photograph, so it carries the
    # frame and can be checked the same way. vis3d itself cannot: it is a
    # rasterization on black, with no photograph in it to disagree with.
    check_channel_order "$OUTPUT_DIR/vis3d_overlay" "$FRAMES_DIR" "vis3d_overlay/" || exit 1
fi

# --- Stage 4: visual odometry ---
# pointcloud is CPU (numpy/opencv/pycocotools); openvo needs the GPU and its
# correlation CUDA extension, which is rebuilt for
# this env rather than used as the py3.9 binary it shipped as -- see
# scripts/vis3d/openvo-correlation-cxx17.patch. estimate_ego_motion.py still
# imports my_inference lazily, so the pointcloud path never touches it.
if [ "$RUN_STAGE4_VO" = 1 ]; then
    python estimate_ego_motion.py --frames_dir $FRAMES_DIR --output_dir $OUTPUT_DIR \
        --scene $VIDEO --method $VO_METHOD

    # Always plot the result. A trajectory is easy to accept on one summary number
    # and wrong in a way only the shape shows: changelane's pointcloud run had a
    # healthy median 35 km/h while its path was a random walk that covered 601 m to
    # get 48 m from the start.
    python plot_ego_trajectory.py --output_dir $OUTPUT_DIR --hz $SOURCE_HZ \
        --title "$VIDEO${RUN:+/$RUN}  $VO_METHOD VO"
fi

# Point maps are only needed to lift boxes/lanes and to run pointcloud VO; both
# are behind us by here, so this is the last chance to drop them.
if [ "$KEEP_DEPTH" != 1 ]; then
    rm -rf "$OUTPUT_DIR/samples-pseudodepth"
    echo "Removed point maps; re-run with RUN_STAGE2_DEPTH=1 if you need to re-lift boxes or lanes."
fi

# --- Stage 5: subsample to the delivery rate ---
# After the point-map cleanup above, so KEEP_DEPTH still targets the run that
# holds them, and before the export, so the navsim log is written from the clip
# that is actually being delivered rather than the one it was computed at.
#
# Skipped entirely when SUBSAMPLE_RUN is empty, which is the single-clip default:
# the run then stays at the rate it was processed at and the clip directory holds
# one run and one frames/ under their own names, rather than two of each plus a
# pair of links. The export below then writes from the processed run at
# SOURCE_HZ, since that is now the clip being delivered.
if [ "$SUBSAMPLE_STRIDE" -gt 1 ] 2>/dev/null && [ -n "$SUBSAMPLE_RUN" ]; then
    python subsample_clip.py --clip_dir "$VIDEO_DIR" --run "$RUN" \
        --frames "$(basename "$FRAMES_DIR")" --stride "$SUBSAMPLE_STRIDE" \
        --out_frames "$SUBSAMPLE_FRAMES" --out_run "$SUBSAMPLE_RUN"
    if [ "$DEBUG_OVERLAY" = 1 ]; then DEBUG_FLAG="--debug"; else DEBUG_FLAG=""; fi
    (cd visualization && python raster_frames.py \
        --output_dir "$VIDEO_DIR/$SUBSAMPLE_RUN" \
        --frames_dir "$VIDEO_DIR/$SUBSAMPLE_FRAMES" \
        --overlay_alpha $OVERLAY_ALPHA $DEBUG_FLAG)

    # Give the delivered clip the names the rest of the tooling expects, but only
    # by adding links and never by replacing anything: a clip that already has a
    # frames/ from an earlier 2 Hz extraction keeps it, and its own run keeps its
    # name. Without this a clip processed this way has no frames/ at all and
    # every glob over the dataset skips it in silence.
    [ -e "$VIDEO_DIR/frames" ] || ln -s "$SUBSAMPLE_FRAMES" "$VIDEO_DIR/frames"
    [ -e "$VIDEO_DIR/1" ]      || ln -s "$SUBSAMPLE_RUN" "$VIDEO_DIR/1"

    # Everything past here delivers the subsampled clip.
    OUTPUT_DIR="$VIDEO_DIR/$SUBSAMPLE_RUN"
    FRAMES_DIR="$VIDEO_DIR/$SUBSAMPLE_FRAMES"
    SOURCE_HZ=$(( SOURCE_HZ / SUBSAMPLE_STRIDE ))
    echo "Delivering at ${SOURCE_HZ} Hz from $OUTPUT_DIR"
fi

# --- navsim-format log, now with ego poses ---
if [ "$EXPORT_NAVSIM" = 1 ]; then
    python export_navsim_logs.py --output_dir $OUTPUT_DIR --frames_dir $FRAMES_DIR \
        --dataset_root "${DATASET_ROOT:-$OUTPUT_DIR}" --split $SPLIT --log_name $VIDEO \
        --source_hz $SOURCE_HZ --poses $OUTPUT_DIR/ego_poses.txt
fi
