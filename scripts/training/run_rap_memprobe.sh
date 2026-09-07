#!/bin/bash

#SBATCH --job-name=rap_memprobe
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

## GAMMA training config
#SBATCH --time=1:00:00
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

# run_training.py:187 hardcodes logger=WandbLogger(...) with no config switch, so wandb is
# not optional -- without a key trainer.fit() dies in the setup hook with
# "UsageError: No API key configured" (observed, job 7410747, after all four caches had
# loaded). offline logs to $WANDB_DIR and needs no login; `wandb sync` uploads later.
# Drop a key in $HOME/.wandb_key to log online instead, mirroring .hf_token above.
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
else
    export WANDB_MODE=offline
fi
export WANDB_DIR=$NAVSIM_EXP_ROOT

# Unlocks all 8 GPUs. Lightning's SLURMEnvironment.world_size() is literally
#     int(os.environ["SLURM_NTASKS"])
# so with `#SBATCH --ntasks=1` DDP came up as "GLOBAL_RANK: 0, MEMBER: 1/1" -- one rank on
# one GPU while the other 7 sat idle, even though Slurm had allocated all 8 (job 7410747:
# gres/gpu:rtxa5000=8). Because the body runs plain `python`, not `srun python`, Slurm never
# creates the other 7 ranks either.
#
# Setting the job name to "bash" is Lightning's own documented escape hatch
# (lightning_fabric/plugins/environments/slurm.py:231, _is_slurm_interactive_mode): it makes
# detect() return False, so Lightning ignores Slurm, uses LightningEnvironment, and spawns
# its own 8 subprocesses over devices="auto" = every visible GPU. This only changes what the
# Python process sees -- squeue and the %x log filename still use the SBATCH --job-name.
#
# The alternative is `srun python` with --ntasks-per-node=8 --cpus-per-task=4; this way is
# a one-liner and leaves the rest of the script alone.
export SLURM_JOB_NAME=bash

# Turns on the _MemProbe callback in run_training.py.
export RAP_MEM_DEBUG=1

# The failing allocations in job 7411178 were 844 MiB and 1.65 GiB while 1.73-1.80 GiB sat
# "reserved but unallocated" -- i.e. the allocator held enough total memory but no single
# block large enough. expandable_segments (torch 2.1+) lets a segment grow in place instead
# of rounding every ViT activation up into its own fixed-size block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $NAVSIM_DEVKIT_ROOT

CACHE_ROOT=$NAVSIM_DEVKIT_ROOT/cache
# Reused from DrivoR: navtrain, 103288 tokens, same train_metric_chache.MetricCache.
# Loading it needs the polygon rebuild in pdm_occupancy_map.py -- swap to
# $NAVSIM_DEVKIT_ROOT/train_metric_cache if you regenerate with run_metric_caching.sh.
METRIC_CACHE=$HOME/RAP/exp/train_metric_cache

# QUICKTEST of run_rap_training.sh: 2 epochs instead of 10
#
# batch_size is PER RANK under DDP, not global: Lightning hands the same dataloader config
# to all 8 subprocesses, so the released default of 64 was really a global batch of 512.
# Each sample carries 4 camera images, and _step_distill concatenates the rendered and real
# halves (agent_lightning_module.py:212) before the single forward, so a per-rank batch of
# B puts 8*B images of 256x1024 through the frozen DINOv3 ViT-H at train time -- 512 at
# B=64, which OOM'd a 24 GB A5000 during the sanity check alone (job 7411178).
# 8 per rank restores the 64 global batch the LR schedule in rap_agent.py was written for.
PRETRAINED=$NAVSIM_DEVKIT_ROOT/weights/RAP_DINO_navsimv2.ckpt
MAX_EPOCHS=2

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
    experiment_name=rap_memprobe \
    train_test_split=navtrain \
    train_test_split/scene_filter=navtrain \
    split=trainval \
    dataset=navsim_dataset \
    cache_path=$CACHE_ROOT/rap_ego \
    cache_path_perturbed=$CACHE_ROOT/rap_perturbed \
    cache_path_others=$CACHE_ROOT/rap_aug \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    dataloader.params.batch_size=8 \
    trainer.params.max_epochs=1 \
    trainer.params.num_sanity_val_steps=0 \
    trainer.params.limit_train_batches=3 \
    trainer.params.limit_val_batches=1 \
    agent.checkpoint_path=$PRETRAINED
