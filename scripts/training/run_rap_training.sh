#!/bin/bash

#SBATCH --job-name=rap_navsim_train
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=240gb
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

## GAMMA training config
#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

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

# run_training.py hardcodes logger=WandbLogger(...) with no config switch, so wandb is
# not optional: without a key trainer.fit() dies in the setup hook with "UsageError: No
# API key configured" -- and it dies late, after every cache has loaded, ~4 minutes in.
# Observed twice, job 7410747 on the quicktest and job 7445728 here. offline logs to
# $WANDB_DIR and needs no login; `wandb sync` uploads afterwards. Drop a key in
# $HOME/.wandb_key to log online instead, mirroring .hf_token above.
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
else
    export WANDB_MODE=offline
fi
# Named explicitly. Left unset, wandb stages artifacts and media into whatever directory
# the job is running in -- the repo root -- under random tmp*wandb-artifacts names.
export WANDB_DIR=$NAVSIM_EXP_ROOT

# Scratch for multiprocessing and torch.distributed. Left to itself this scattered 32
# empty pymp-* directories and assorted tmp* through the checkout (jobs 7435020/7435526/
# 7435545, all on gammagpu04). Why it landed in the repo is NOT established: TMPDIR is
# inherited as /tmp on both gammagpu01 and legacy02, /tmp is writable and xfs on both,
# and nothing in navsim/ or rl/ touches tempfile.tempdir. gammagpu04 evidently differs
# in some way that has not been characterised.
#
# So this does not assume either answer. Node-local /tmp is preferred when it is usable
# -- it is xfs, so the NFS silly-rename that makes every dataloader-worker teardown
# print an "OSError: Device or resource busy: .nfsXXXX" traceback cannot happen there,
# and the node clears it without help. The project filesystem is the fallback, which is
# correct but on NFS, so it keeps that harmless noise.
_scratch=/tmp/rap_${SLURM_JOB_ID:-manual}
if mkdir -p "$_scratch" 2>/dev/null && [ -w "$_scratch" ]; then
    export TMPDIR=$_scratch
else
    export TMPDIR=$NAVSIM_DEVKIT_ROOT/tmp/${SLURM_JOB_ID:-manual}
    mkdir -p "$TMPDIR"
    echo "note: /tmp unusable on $(hostname); scratch on NFS at $TMPDIR"
fi
echo "TMPDIR=$TMPDIR"
# Best effort. A SIGKILL -- an OOM, a walltime hit -- never runs this, which is why the
# directory carries the job id rather than random characters: whatever survives is
# attributable to the run that left it.
trap '[ -n "$TMPDIR" ] && rm -rf "$TMPDIR"' EXIT

cd $NAVSIM_DEVKIT_ROOT

CACHE_ROOT=$NAVSIM_DEVKIT_ROOT/cache
# Reused from DrivoR: navtrain, 103288 tokens, same train_metric_chache.MetricCache.
# Loading it needs the polygon rebuild in pdm_occupancy_map.py -- swap to
# $NAVSIM_DEVKIT_ROOT/train_metric_cache if you regenerate with run_metric_caching.sh.
METRIC_CACHE=$HOME/RAP/exp/train_metric_cache

# Warm start from the authors' NAVSIM v2 release (weights only -- init_from_pretrained
# loads state_dict with strict=False, so optimizer state and the epoch counter reset).
# The cosine period at rap_agent.py:585 was set to 10 to match MAX_EPOCHS below, so the
# LR anneals 1.0e-4 -> 1.27e-5 across the run instead of stopping mid-decay.
PRETRAINED=$NAVSIM_DEVKIT_ROOT/weights/RAP_DINO_navsimv2.ckpt
MAX_EPOCHS=10

# PER RANK under DDP, not global: Lightning hands the same dataloader config to all 8
# subprocesses, so the released default of 64 was really a global batch of 512. Each
# sample carries 4 camera images and _step_distill concatenates the rendered and real
# halves before the single forward, so a per-rank batch of B pushes 8*B images of
# 256x1024 through the frozen DINOv3 ViT-H at train time.
#
# 8 restores the 64 global batch the LR schedule in rap_agent.get_optimizers was written
# for. It is also the measured ceiling: B=16 cleared the sanity check and OOM'd on the
# first training step (job 7418298), and B=64 did the same here (job 7445866, at
# spatial_cross_attention.py:128). That block casts q/k/v to fp32 under
# torch.autocast(enabled=False), so the heaviest trainable module gets no benefit from
# mixed precision and holds fp32 activations for backward.
#
# Note validation is half the forward width of training: clearing the sanity check says
# nothing about whether the training step fits. Both OOMs above got past it.
BATCH_SIZE=8

# Checkpoint every other epoch and keep all of them: with MAX_EPOCHS=10 that is 5
# files at epochs 1,3,5,7,9 (Lightning counts from 0), roughly 4 GB each, ~20 GB.
# KEEP_N=-1 is what makes them accumulate -- at the default of 1 each write would
# replace the previous file and a wider interval would only mean checkpointing
# less often, not keeping a history.
#
# Note the resume cost: run_training.py fits with ckpt_path='last', which lands on
# the newest file here, so a job killed at epoch 4 restarts from epoch 3 and repeats
# one epoch. That is the trade for a 2-epoch interval; set it to 1 if the wall clock
# matters more than the disk.
CKPT_EVERY_N_EPOCHS=2
CKPT_KEEP_N=-1

if [ ! -f "$PRETRAINED" ]; then
    echo "Missing pretrained checkpoint $PRETRAINED"
    exit 1
fi

for d in $CACHE_ROOT/rap_ego $CACHE_ROOT/rap_aug $CACHE_ROOT/rap_perturbed $METRIC_CACHE; do
    if [ ! -d "$d" ]; then
        echo "Missing cache $d -- run scripts/training/run_dataset_caching.sh first."
        exit 1
    fi
done

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=rap_agent \
    agent.config.pdm_scorer=True \
    agent.config.distill_feature=True \
    agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.train_metric_cache_path=$METRIC_CACHE \
    experiment_name=rap_navsim_train \
    train_test_split=navtrain \
    train_test_split/scene_filter=navtrain \
    split=trainval \
    dataset=navsim_dataset \
    cache_path=$CACHE_ROOT/rap_ego \
    cache_path_perturbed=$CACHE_ROOT/rap_perturbed \
    cache_path_others=$CACHE_ROOT/rap_aug \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    dataloader.params.batch_size=$BATCH_SIZE \
    trainer.params.max_epochs=$MAX_EPOCHS \
    agent.config.checkpoint_every_n_epochs=$CKPT_EVERY_N_EPOCHS \
    agent.config.checkpoint_keep_n=$CKPT_KEEP_N \
    agent.checkpoint_path=$PRETRAINED
