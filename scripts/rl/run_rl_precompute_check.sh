#!/bin/bash

#SBATCH --job-name=rap_rl_check
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

# Preflight for the RL pipeline. Minutes, not hours -- run it before committing a GPU to
# run_rl_precompute.sh, and again after touching precompute.py, agent.py or residual.py.
#
#   1. rl/parity_check.py -- the observation the benchmark agent builds is elementwise
#      identical to the one precompute.py writes. Drift here is invisible at run time:
#      the policy silently receives an input it was never trained on, and the benchmark
#      reports that as "RL did not help".
#   2. a 32-token precompute into a throwaway directory, proving the encode + PDM
#      scoring path works end to end before it is asked to do 14675 of them.

# Absolute, not $(dirname "$0"): sbatch copies this script into a spool directory
# (/var/spool/slurm/.../slurm_script), so $0 does not resolve to the repo and the
# source silently fails there while working fine when run by hand. The #SBATCH
# output paths above are hardcoded for the same reason.
source /fs/nexus-projects/sim2real/aliu/RAP/scripts/rl/env.sh
cd $NAVSIM_DEVKIT_ROOT

SMOKE_CACHE=$NAVSIM_DEVKIT_ROOT/cache/rl_obs_smoke

echo "=============== 1/2 observation parity ==============="
python rl/parity_check.py --limit 8 --batch-size 4 --time-horizon 5 || exit 1

echo
echo "=============== 2/2 precompute smoke ==============="
# Its own directory, never cache/rl_obs: precompute skips shards that already exist, so
# a 32-token shard left in the real cache would be silently adopted by the full run.
rm -rf "$SMOKE_CACHE"
python rl/precompute.py \
    --limit 32 --batch-size 4 --shard-size 32 \
    --num-workers 4 --score-workers 8 --time-horizon 5 \
    --rl-cache-path "$SMOKE_CACHE" || exit 1

echo
python - <<'PYEOF'
import os, sys
import numpy as np
sys.path.insert(0, os.environ["NAVSIM_DEVKIT_ROOT"])
from pathlib import Path
from rl.env import RLDataset

d = RLDataset(Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "cache" / "rl_obs_smoke")
print(f"[smoke] {len(d)} tokens, {d.num_poses} poses/trajectory")
print(f"[smoke] latent {d.scene_latent.shape[1:]}, ego {d.ego_status.shape[1:]}")
print(f"[smoke] base PDMS {np.nanmean(d.base_pdm[:, -1]):.4f}  "
      f"collision rate {1.0 - np.nanmean(d.base_pdm[:, 0]):.4f}")
finite = int(np.isfinite(d.base_pdm).all(1).sum())
print(f"[smoke] finite baselines: {finite}/{len(d)}")
# A cache whose baselines are all NaN trains against nothing; rl/env.py would refuse it
# anyway, but failing here costs seconds instead of a queued GPU-hour.
raise SystemExit(0 if finite else 1)
PYEOF
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
    echo "PREFLIGHT PASSED -- safe to run scripts/rl/run_rl_precompute.sh"
else
    echo "PREFLIGHT FAILED"
fi
exit $STATUS
