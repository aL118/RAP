#!/bin/bash

#SBATCH --job-name=rap_iterative_quicktest
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

## ONE GPU, unlike run_rap_training_quicktest.sh's eight. Nothing in rl/ is
## distributed: collect.py and train.py are plain single-process scripts that
## take `--device`, there is no Lightning and no DDP, so a second GPU would sit
## idle for the whole run. That also makes the two workarounds in the supervised
## script unnecessary here and they are deliberately absent:
##   * `export SLURM_JOB_NAME=bash` is a Lightning SLURMEnvironment escape hatch.
##     No Lightning, nothing to escape.
##   * WANDB_MODE=offline is there because run_training.py hardcodes WandbLogger.
##     rl/train.py logs to stdout.
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
## Sized for the scoring pool, not the GPU: score_workers below is a *process*
## count (rl/scoring.py's PDMScorer keeps per-call state on a module-level
## singleton, so threads would silently corrupt each other's labels), and each
## spawned worker imports nuplan afresh.
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb

#SBATCH --time=8:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate rap

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/RAP"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# rl/config.py reads this for the navtrain half's PDM metric cache. Same cache
# the supervised quicktest uses, reused from DrivoR via the exp/ symlink; it is
# agent-independent, so any trajectory can be scored against it after the fact.
export RAP_METRIC_CACHE=$NAVSIM_DEVKIT_ROOT/exp/train_metric_cache

# Same reason as the supervised script: the frozen ViT's activations are large
# and unevenly sized, so let a segment grow in place rather than rounding every
# one up into its own block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $NAVSIM_DEVKIT_ROOT

# Its own experiment name, and not the one the CPU smoke test used. The buffer
# lives inside the run directory because it *is* the run -- and the smoke
# buffer's rows were scored against the PREVIOUS CARE export, whose windows are
# a different set of scenes. Training on a mix of the two would be training on
# labels for footage that is no longer there.
EXPERIMENT=rap_iterative_quicktest

# Round 1 samples from the pretrained planner by definition (rl/config.py:
# round_checkpoint(0) IS checkpoint_path), so "5 epochs from the pretrained
# model" is one round at 5 epochs. num_rounds=1 stops there rather than going on
# to sample from what round 1 produced.
MAX_EPOCHS=5

# Per-scene batch, not per rank -- there is only one process. A CARE window
# carries 1 camera and a navtrain sample carries 4, and rl/data.py never mixes
# them in a batch, so the widest forward here is 4*4=16 images through the ViT.
# The supervised script's B=8 pushed 8*4*2=64 (it has the distill path doubling
# the batch; rl/ does not, distill_feature is False).
BATCH_SIZE=4
SCORE_WORKERS=12

PRETRAINED=$NAVSIM_DEVKIT_ROOT/weights/RAP_DINO_navsimv2.ckpt
if [ ! -f "$PRETRAINED" ]; then
    echo "Missing pretrained checkpoint $PRETRAINED"
    exit 1
fi
if [ ! -d "$NAVSIM_DEVKIT_ROOT/CARE/openscene_meta_datas" ]; then
    echo "Missing CARE export -- run scripts/vis3d/generate_ras_logs.py first."
    exit 1
fi
for d in $NAVSIM_DEVKIT_ROOT/cache/rap_ego $RAP_METRIC_CACHE; do
    if [ ! -d "$d" ]; then
        echo "Missing cache $d -- needed for the navtrain half (regular_ratio > 0)."
        exit 1
    fi
done

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# collect -> train, resumable at round granularity: the buffer file and the
# checkpoint are each written once and atomically, so relaunching this script
# after a failure skips whichever half already landed rather than rescoring it.
python $NAVSIM_DEVKIT_ROOT/rl/loop.py \
    --num-rounds 1 \
    --epochs-per-round $MAX_EPOCHS \
    --batch-size $BATCH_SIZE \
    --score-workers $SCORE_WORKERS \
    --experiment-name $EXPERIMENT \
    --device cuda

# Both stages write per-batch scalars to $NAVSIM_EXP_ROOT/$EXPERIMENT/logs/*.jsonl
# and TensorBoard event files under .../tb/. The JSONL is line-buffered, so
# `tail -f` and `jq` follow this job from a login node without a port forward --
# see "Watching a run" in rl/README.md. --no-tensorboard drops the event files.
echo "[run] curves: tensorboard --logdir $NAVSIM_EXP_ROOT/$EXPERIMENT/tb"

# --baseline evaluates round 0 as well, so the delta comes from one code path on
# one split. Neither number is comparable to a published NAVSIM score: the split
# is a held-out slice of the training sources and half of it is CARE footage
# with no drivable-area label. See rl/README.md for what the columns mean --
# `selected` is the headline, and `pool_best` is not a ceiling.
python $NAVSIM_DEVKIT_ROOT/rl/eval.py \
    --baseline \
    --round 1 \
    --limit 96 \
    --proposals-per-scene 16 \
    --score-workers $SCORE_WORKERS \
    --experiment-name $EXPERIMENT \
    --device cuda
