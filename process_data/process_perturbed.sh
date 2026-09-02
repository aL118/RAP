#!/bin/bash

#SBATCH --job-name=rap_process_perturbed
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

# Step 2 of process.sh ONLY, re-run to add the CAM_B0 renders that the first pass
# never produced: _purturbed.py:293 used to call ScenarioRenderer() bare, and
# helpers/renderer.py:695 defaults to ['CAM_F0', 'CAM_L0', 'CAM_R0'].
#
# Why that mattered: rap_agent.py:130-142 requests cam_b0, and
# rap_dino/bevformer/bev_feature_build.py:29 feeds cameras.cam_b0 into the synthetic
# branch FIRST with validity hardcoded True -- while dataclasses.py:80-82 catches the
# missing JPG with a bare except and substitutes np.zeros((1080,1920,3)). A perturbed
# cache built off the old renders therefore carries a black rear view on every token,
# marked valid, and nothing downstream reports it.
#
# Same shape as process.sh: no GPU, rendering is cv2 on numpy, ~8 GB per worker.
#SBATCH --mem=120gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=12:00:00
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
export NUPLAN_PATH="/fs/nexus-projects/sim2real/aliu/nuplan/dataset/nuplan-v1.1"
export NUPLAN_DB_PATH=${NUPLAN_PATH}/splits/${split}
# --------------------------- UPDATE THIS ---------------------------

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"

# helpers/nuplan_cameras_utils.py:23 reads this at import time (KeyError without it).
export NUPLAN_SENSOR_PATH="$HOME/navsim/dataset/sensor_blobs/$split"

DATA_PERTURBED=$HOME/navsim/dataset_perturbed

# OUTPUT template: the script replaces sensor_blobs -> rendered_sensor_blobs in it to
# pick where renders go. Nothing is read from it.
SENSOR_TEMPLATE_PERTURBED=$DATA_PERTURBED/sensor_blobs/$split

if [ ! -d "$NUPLAN_DB_PATH" ]; then
    echo "FATAL: NUPLAN_DB_PATH does not exist: $NUPLAN_DB_PATH"
    exit 1
fi

cd $NAVSIM_DEVKIT_ROOT/process_data

# The whole point of this run is to redo work the checkpoint records as done, so the
# 287 completed indices in checkpoint_perturbed.txt have to be out of the way -- with
# them in place load_done_set() (_purturbed.py:586) skips every log and this exits
# having changed nothing. Kept, not deleted: it is the only record of the first pass.
STAMP=$(date +%Y%m%d_%H%M%S)
if [ -f checkpoint_perturbed.txt ]; then
    mv checkpoint_perturbed.txt "checkpoint_perturbed.txt.pre_b0.$STAMP"
    echo "moved checkpoint_perturbed.txt -> checkpoint_perturbed.txt.pre_b0.$STAMP"
fi
rm -f checkpoint.txt

# Outputs are overwritten in place, not appended to: both the .pkl stem and the JPG
# name derive from lidar_pc.token out of the nuPlan DB (_purturbed.py:332), not from a
# hash of the perturbed pose, so the 22,234 stems stay identical across runs and no
# orphans are left behind. The random.uniform() perturbations at _purturbed.py:248-261
# are unseeded, so the pose VALUES will differ from the first pass -- each pkl and its
# four renders are regenerated together, so they stay consistent with each other.
OUT_DIR="$DATA_PERTURBED/navsim_logs/$split"
mkdir -p "$OUT_DIR"

python -u create_openscene_metadata_purturbed.py \
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

mv -f checkpoint.txt checkpoint_perturbed.txt 2>/dev/null || true

# Verify the thing this run exists to fix, rather than trusting it. A silent partial
# render is exactly the failure mode that made the re-run necessary.
RENDER_ROOT="$DATA_PERTURBED/rendered_sensor_blobs/$split"
logs=$(ls "$RENDER_ROOT" | wc -l)
with_b0=$(ls -d "$RENDER_ROOT"/*/CAM_B0 2>/dev/null | wc -l)
echo
echo "---- $RENDER_ROOT"
echo "  log dirs        : $logs"
echo "  with CAM_B0     : $with_b0"
echo "  total JPGs      : $(find "$RENDER_ROOT" -name '*.jpg' | wc -l)"
echo "  pkls            : $(find "$OUT_DIR" -maxdepth 1 -name '*.pkl' | wc -l)"
if [ "$with_b0" -ne "$logs" ]; then
    echo "FATAL: $((logs - with_b0)) log dirs still have no CAM_B0."
    exit 1
fi
echo
echo "CAM_B0 present for every log. Next:"
echo "  python scripts/training/make_cache_subsets_rap.py --dataset perturbed --budget-gb <N>"
echo "  sbatch scripts/data/run_dataset_caching_perturbed.sh"
