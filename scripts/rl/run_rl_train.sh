#!/bin/bash

#SBATCH --job-name=rap_rl_ppo
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# One GPU, many CPUs. The policy is a small MLP -- the bottleneck is the PDM scorer,
# which is pure CPU at ~0.2 s per scored trajectory. n_envs is therefore a CPU count,
# not a GPU count; 16 workers give roughly 80 env steps/s.
#
# Each SubprocVecEnv worker holds its own ~300 MB copy of the observation cache plus
# the metric-cache index, hence the memory request.

# Absolute, not $(dirname "$0"): sbatch copies this script into a spool directory
# (/var/spool/slurm/.../slurm_script), so $0 does not resolve to the repo and the
# source silently fails there while working fine when run by hand. The #SBATCH
# output paths above are hardcoded for the same reason.
source /fs/nexus-projects/sim2real/aliu/RAP/scripts/rl/env.sh
cd $NAVSIM_DEVKIT_ROOT

if [ ! -d "$NAVSIM_DEVKIT_ROOT/cache/rl_obs" ]; then
    echo "Missing cache/rl_obs -- run scripts/rl/run_rl_precompute.sh first."
    exit 1
fi

python rl/train.py \
    --n-envs 16 \
    --n-steps 64 \
    --batch-size 256 \
    --total-timesteps 200000 \
    --experiment-name rap_rl_ppo

python rl/eval.py --experiment-name rap_rl_ppo --n-episodes 400
