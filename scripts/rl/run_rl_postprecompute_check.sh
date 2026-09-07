#!/bin/bash

#SBATCH --job-name=rap_rl_post
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# Run after the full precompute, before committing a GPU-day to training.
#
#   1. the --probe calibration table: which perturbation directions the PDM reward
#      actually responds to. residual_scale is sized from this; a column that reads
#      "flat" is a direction the policy can never get gradient from.
#   2. the residual path end to end, using whatever policy is passed in POLICY. This
#      exercises the parts run_rl_precompute_check.sh cannot: the action bound, the
#      ramp, VecNormalize, and the obs_meta.json provenance check. The policy's
#      *numbers* are irrelevant here -- this is a plumbing test, and a stale policy
#      is fine for it.

source /fs/nexus-projects/sim2real/aliu/RAP/scripts/rl/env.sh
cd $NAVSIM_DEVKIT_ROOT

echo "=============== 1/2 reward sensitivity probe ==============="
python rl/eval.py --probe --probe-scenes ${PROBE_SCENES:-40}

POLICY=${POLICY:-$NAVSIM_DEVKIT_ROOT/exp/rl_smoke/final_model.zip}
if [ -f "$POLICY" ]; then
    echo
    echo "=============== 2/2 residual path (policy: $POLICY) ==============="
    echo "NOTE: a plumbing test. Only the bound/ramp/OK lines mean anything;"
    echo "      the trajectory values depend on whichever policy was passed."
    python rl/parity_check.py --limit 4 --batch-size 4 --time-horizon 5 \
        --policy-path "$POLICY"
else
    echo "No policy at $POLICY -- skipping the residual path check."
fi
