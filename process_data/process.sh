#!/bin/bash

#SBATCH --job-name=rap_process_data
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

# No GPU: rendering is cv2 on numpy. Memory is per --thread-num worker (~8 GB each).
#SBATCH --mem=120gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

eval "$(conda shell.bash hook)"
conda activate rap

export HOME="/fs/nexus-projects/sim2real/aliu"
export NAVSIM_DEVKIT_ROOT="$HOME/RAP"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

split=trainval
THREADS=32

# --------------------------- UPDATE THIS ---------------------------
# Raw nuPlan v1.1 from https://www.nuscenes.org/nuplan. Only the .db logs are read.
export NUPLAN_PATH="/fs/nexus-projects/sim2real/aliu/nuplan/dataset/nuplan-v1.1"
export NUPLAN_DB_PATH=${NUPLAN_PATH}/splits/${split}
# --------------------------- UPDATE THIS ---------------------------

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"

# helpers/nuplan_cameras_utils.py:23 reads this at import time (KeyError without it).
# It is the REAL nuPlan camera JPGs, used only to test whether each image exists on
# disk -- unrelated to the --nuplan-sensor-path OUTPUT templates below.
export NUPLAN_SENSOR_PATH="$HOME/navsim/dataset/sensor_blobs/$split"

DATA_PERTURBED=$HOME/navsim/dataset_perturbed
DATA_AUG=$HOME/navsim/dataset_aug

# --nuplan-sensor-path is an OUTPUT template: both scripts replace sensor_blobs ->
# rendered_sensor_blobs[_augmented] in it to pick where renders go. Nothing is read from it.
SENSOR_TEMPLATE_PERTURBED=$DATA_PERTURBED/sensor_blobs/$split
SENSOR_TEMPLATE_AUG=$DATA_AUG/sensor_blobs/$split

if [ ! -d "$NUPLAN_DB_PATH" ]; then
    echo "FATAL: NUPLAN_DB_PATH does not exist: $NUPLAN_DB_PATH"
    exit 1
fi

cd $NAVSIM_DEVKIT_ROOT/process_data

# _aug.py writes to rendered_sensor_blobs_augmented; navsim reads rendered_sensor_blobs.
link_augmented_renders () {
    local root=$1
    rmdir "$root/rendered_sensor_blobs/$split" "$root/rendered_sensor_blobs" 2>/dev/null || true
    if [ -d "$root/rendered_sensor_blobs_augmented" ] && [ ! -e "$root/rendered_sensor_blobs" ]; then
        ln -sfn rendered_sensor_blobs_augmented "$root/rendered_sensor_blobs"
    fi
}

# All three scripts hardcode checkpoint.txt in cwd (_purturbed.py:580, _aug.py:918) and
# store POSITIONAL indices into sorted(os.listdir(NUPLAN_DB_PATH)). Shared between steps it
# makes step 3 skip every log step 2 finished, so each step gets its own copy. The indices
# are only valid while NUPLAN_DB_PATH holds the same files: delete these when you swap in a
# different batch of .db logs.
run_step () {
    local tag=$1; shift
    [ -f "checkpoint_$tag.txt" ] && cp "checkpoint_$tag.txt" checkpoint.txt || rm -f checkpoint.txt
    "$@"
    mv -f checkpoint.txt "checkpoint_$tag.txt" 2>/dev/null || true
}

OUT_DIR="$HOME/navsim/dataset/navsim_logs/$split"
# 1. OpenScene metadata + rasterized views. Already done; cache/rap_ego is built from it.
# python -u create_openscene_metadata.py \
#   --nuplan-root-path ${NUPLAN_PATH} \
#   --nuplan-db-path ${NUPLAN_DB_PATH} \
#   --nuplan-sensor-path $HOME/navsim/dataset/sensor_blobs/${split} \
#   --nuplan-map-version nuplan-maps-v1.0 \
#   --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
#   --out-dir ${OUT_DIR} \
#   --split ${split} \
#   --thread-num ${THREADS} \
#   --start-index 0 \
#   --end-index 14561

OUT_DIR="$DATA_PERTURBED/navsim_logs/$split"
mkdir -p "$OUT_DIR"
# 2. Recovery-oriented trajectory perturbations. Filename is _purturbed upstream.
run_step perturbed python -u create_openscene_metadata_purturbed.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${SENSOR_TEMPLATE_PERTURBED} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num ${THREADS} \
  --start-index 0 \
  --end-index 14561

OUT_DIR="$DATA_AUG/navsim_logs/$split"
# _aug.py:878 opens its .pkl without creating out_dir first -- _purturbed.py:540 does
# (os.makedirs(args.out_dir, exist_ok=True)), _aug.py has no equivalent anywhere, so
# step 3 dies with FileNotFoundError on its first write. Create it here.
mkdir -p "$OUT_DIR"
# 3. Cross-agent view synthesis
run_step aug python -u create_openscene_metadata_aug.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${SENSOR_TEMPLATE_AUG} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num ${THREADS} \
  --start-index 0 \
  --end-index 14561

link_augmented_renders "$DATA_AUG"
