#!/bin/bash

#SBATCH --job-name=rap_rl_smoke_e2e
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# End-to-end plumbing test: precompute -> train -> eval on a small slice, sized to
# finish inside the 1 h walltime above. This answers "does the pipeline run start to
# finish", NOT "does RL help" -- 6k timesteps on 300 tokens teaches the policy nothing
# and the eval delta is noise. Read the stage banners and the exit code, not the PDMS.
#
# Absolute, not $(dirname "$0"): sbatch copies this script into a spool directory
# (/var/spool/slurm/.../slurm_script), so $0 does not resolve to the repo and the
# source silently fails there while working fine when run by hand. The #SBATCH
# output paths above are hardcoded for the same reason.
source /fs/nexus-projects/sim2real/aliu/RAP/scripts/rl/env.sh
cd $NAVSIM_DEVKIT_ROOT

# A separate cache dir, never cache/rl_obs. precompute.py resumes by skipping shards
# that already exist, so a 400-token smoke cache left in the real location would be
# silently adopted by the next full run as if it were complete.
SMOKE_CACHE="$NAVSIM_DEVKIT_ROOT/cache/rl_obs_smoke"
EXP_NAME=rap_rl_smoke_e2e

# Start clean: exercising the shard write path is the point, and a leftover shard from
# an earlier smoke would be skipped instead of rewritten. Guarded so a mistyped edit
# to SMOKE_CACHE cannot delete the real cache.
case "$SMOKE_CACHE" in
    *rl_obs_smoke) rm -rf "$SMOKE_CACHE" ;;
    *) echo "refusing to clear unexpected path $SMOKE_CACHE"; exit 1 ;;
esac

set -e
trap 'echo "!!! smoke test FAILED at stage $STAGE"' ERR

# 400 tokens over 2 shards of 200: two trips through the shard loop, so the tmp-write
# and the exists-and-readable resume branch both get exercised, not just the first.
STAGE="1/3 precompute"
echo "=============== $STAGE ==============="
python rl/precompute.py \
    --batch-size 8 \
    --num-workers 12 \
    --shard-size 200 \
    --limit 400 \
    --time-horizon 5 \
    --rl-cache-path "$SMOKE_CACHE"

# val-fraction 0.25, not the 0.05 default: 5% of 400 tokens is a 20-scene held-out set,
# too small for the eval stage to say anything at all. Train and eval must be given the
# SAME value -- the split is a hash of the token, so they only agree if the fraction does.
STAGE="2/3 train"
echo "=============== $STAGE ==============="
python rl/train.py \
    --n-envs 12 \
    --n-steps 64 \
    --batch-size 128 \
    --total-timesteps 6000 \
    --eval-freq 3000 \
    --val-fraction 0.25 \
    --rl-cache-path "$SMOKE_CACHE" \
    --experiment-name "$EXP_NAME"

STAGE="3/3 eval"
echo "=============== $STAGE ==============="
python rl/eval.py \
    --n-episodes 50 \
    --val-fraction 0.25 \
    --rl-cache-path "$SMOKE_CACHE" \
    --experiment-name "$EXP_NAME"

echo
echo "=============== smoke test PASSED (all 3 stages) ==============="
