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

############################### CONFIG ###############################
BASE=/fs/nexus-projects/sim2real/aliu/RAP
DATA_ROOT="$BASE/data"           # searched recursively for $BOXES
# Which clips to render. Empty = every clip under $DATA_ROOT that has $BOXES,
# which is what this script has always done. Name clips to render only those:
#
#   CLIPS=hydroplaning ./vis3d.sh
#   CLIPS="roundabout merge_in" DATASET=CARE_YTB ./vis3d.sh
#   CLIPS=/fs/.../data/CARE_YTB/hydroplaning ./vis3d.sh   # a path works too
#
# A bare name is looked up as <data>/<dataset>/<name>; set DATASET to say which
# dataset, or leave it empty to take the name from any of them (and be told if
# it is ambiguous). A named clip that turns up nothing is an error rather than a
# silent no-op -- asking for one clip and rendering none is always a typo.
CLIPS="${CLIPS:-}"
DATASET="${DATASET:-}"
# The boxes file to render. gemini_boxes_3d.json is the reviewed output and the
# default; clips that have never been through the reviewer only have
# boxes_3d.json, and are invisible to a default run:
#
#   CLIPS=hydroplaning BOXES=boxes_3d.json ./vis3d.sh
BOXES="${BOXES:-gemini_boxes_3d.json}"
VIS_SUBDIR=vis3d                 # overlay dir is this + '_overlay'
OVERLAY_ALPHA=0.6
# gemini_vis3d/ is what this script replaces: same boxes, same frames, second
# copy, so it is pure duplicate once the swap below lands. Off by default
# anyway -- this script's remit is vis3d/ and vis3d_overlay/, and deleting a
# directory nobody asked it to touch is not its call. Set to 1 to reclaim it.
PRUNE_GEMINI_DIRS=0
DRY_RUN=${DRY_RUN:-0}
######################################################################

eval "$(conda shell.bash hook)"
# `set +u` around the activation because conda's own activation scripts read
# unset variables (binutils' ADDR2LINE and friends), fatal under `set -u`.
set +u
conda activate vis3d
set -u

# raster_frames.py imports renderer.py locally, so run it from visualization/.
cd "$BASE/vis3d/visualization"

# Resolve a clip name to its directory: a path is taken as given, a bare name is
# looked up under $DATASET, or across every dataset when DATASET is empty. A
# name that matches in two datasets is an error -- picking one silently renders
# the wrong clip.
resolve_clip() {
    local name="$1" hits=()
    if [ -d "$name" ]; then
        printf '%s\n' "$name"; return 0
    fi
    if [ -n "$DATASET" ]; then
        [ -d "$DATA_ROOT/$DATASET/$name" ] || return 1
        printf '%s\n' "$DATA_ROOT/$DATASET/$name"; return 0
    fi
    mapfile -t hits < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d \
                             -name "$name" | sort)
    case "${#hits[@]}" in
        0) return 1 ;;
        1) printf '%s\n' "${hits[0]}"; return 0 ;;
        *) echo "!! $name is in ${#hits[@]} datasets: ${hits[*]}" >&2
           echo "   set DATASET= to say which." >&2
           return 2 ;;
    esac
}

BOX_FILES=()
if [ -z "$CLIPS" ]; then
    SEARCH_ROOT="$DATA_ROOT${DATASET:+/$DATASET}"
    mapfile -t BOX_FILES < <(find "$SEARCH_ROOT" -name "$BOXES" | sort)
    if [ "${#BOX_FILES[@]}" -eq 0 ]; then
        echo "No $BOXES anywhere under $SEARCH_ROOT -- nothing to render." >&2
        exit 1
    fi
    echo "Found ${#BOX_FILES[@]} clips with $BOXES."
else
    # Named clips. Each is searched the same way the sweep is, so a clip with
    # several run dirs still renders all of them.
    for clip in $CLIPS; do
        if ! clip_dir=$(resolve_clip "$clip"); then
            echo "!! $clip: no such clip${DATASET:+ in $DATASET} under $DATA_ROOT" >&2
            exit 1
        fi
        mapfile -t found < <(find "$clip_dir" -name "$BOXES" | sort)
        if [ "${#found[@]}" -eq 0 ]; then
            echo "!! $clip: no $BOXES under $clip_dir" >&2
            echo "   (BOXES=boxes_3d.json renders a clip the reviewer never saw.)" >&2
            exit 1
        fi
        BOX_FILES+=("${found[@]}")
    done
    echo "Rendering ${#BOX_FILES[@]} run(s) from $BOXES."
fi

FAILED=()
SKIPPED=()
for BOX_FILE in "${BOX_FILES[@]}"; do
    OUTPUT_DIR=$(dirname "$BOX_FILE")       # .../<clip>/<run>
    CLIP_DIR=$(dirname "$OUTPUT_DIR")       # .../<clip>
    FRAMES_DIR="$CLIP_DIR/frames"
    REL=${OUTPUT_DIR#"$DATA_ROOT"/}

    # No frames means no overlay and no frame list to render against -- the
    # boxes file alone only names the frames something was detected in.
    if [ ! -d "$FRAMES_DIR" ]; then
        echo "!! $REL: no $FRAMES_DIR, skipping."
        SKIPPED+=("$REL")
        continue
    fi

    NEW_SUBDIR="${VIS_SUBDIR}.new"
    if [ "$DRY_RUN" = 1 ]; then
        echo ">> $REL: would render $(ls "$FRAMES_DIR" | wc -l) frames over" \
             "$VIS_SUBDIR/ + ${VIS_SUBDIR}_overlay/"
        if [ "$PRUNE_GEMINI_DIRS" = 1 ] && [ -d "$OUTPUT_DIR/gemini_$VIS_SUBDIR" ]; then
            echo "   ... and delete gemini_$VIS_SUBDIR/ + gemini_${VIS_SUBDIR}_overlay/"
        fi
        continue
    fi

    echo ">> $REL"
    rm -rf "$OUTPUT_DIR/$NEW_SUBDIR" "$OUTPUT_DIR/${NEW_SUBDIR}_overlay"
    if ! python raster_frames.py \
            --output_dir "$OUTPUT_DIR" \
            --frames_dir "$FRAMES_DIR" \
            --boxes "$BOXES" \
            --vis_subdir "$NEW_SUBDIR" \
            --overlay_alpha "$OVERLAY_ALPHA"; then
        echo "!! $REL: raster_frames.py failed, keeping the existing render."
        rm -rf "$OUTPUT_DIR/$NEW_SUBDIR" "$OUTPUT_DIR/${NEW_SUBDIR}_overlay"
        FAILED+=("$REL")
        continue
    fi

    # Swap: the old render only goes once the new one is on disk.
    rm -rf "$OUTPUT_DIR/$VIS_SUBDIR" "$OUTPUT_DIR/${VIS_SUBDIR}_overlay"
    mv "$OUTPUT_DIR/$NEW_SUBDIR" "$OUTPUT_DIR/$VIS_SUBDIR"
    mv "$OUTPUT_DIR/${NEW_SUBDIR}_overlay" "$OUTPUT_DIR/${VIS_SUBDIR}_overlay"
    if [ "$PRUNE_GEMINI_DIRS" = 1 ]; then
        rm -rf "$OUTPUT_DIR/gemini_$VIS_SUBDIR" "$OUTPUT_DIR/gemini_${VIS_SUBDIR}_overlay"
    fi
done

if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "Skipped (no frames/): ${SKIPPED[*]}"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "Done."
