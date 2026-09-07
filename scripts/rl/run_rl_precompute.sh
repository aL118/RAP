#!/bin/bash

#SBATCH --job-name=rap_rl_precompute
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# One GPU is enough: this is inference only, and the run is resumable at shard
# granularity, so an interrupted job is restarted by resubmitting this script.

# Absolute, not $(dirname "$0"): sbatch copies this script into a spool directory
# (/var/spool/slurm/.../slurm_script), so $0 does not resolve to the repo and the
# source silently fails there while working fine when run by hand. The #SBATCH
# output paths above are hardcoded for the same reason.
source /fs/nexus-projects/sim2real/aliu/RAP/scripts/rl/env.sh
cd $NAVSIM_DEVKIT_ROOT

# batch_size=8 -> 32 images of 448x768 through the frozen DINOv3 ViT-H per forward.
# Inference has no gradient buffers, so this sits far under the 24 GB the supervised
# trainer needs at the same batch size.
python rl/precompute.py \
    --batch-size 8 \
    --num-workers 12 \
    --shard-size 2000 \
    --time-horizon 5
