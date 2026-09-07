#!/bin/bash

#SBATCH --job-name=rap_rl_navhard
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# Benchmark the RL-fine-tuned planner the same way the baseline was benchmarked.
#
# rl/eval.py answers a different question: it replays the held-out slice of the
# precomputed navtrain cache, reusing the frozen latents and the frozen base
# trajectories the policy trained on. Useful during training, not comparable to
# anything published. This script runs the devkit's run_pdm_score.py over
# navhard_two_stage from raw sensors, which is what produced the baseline number --
# so the only thing that differs between the two columns is the residual.
#
# Read scripts/evaluation/run_navhard_evaluation.sh first: it explains why this runs
# the devkit rather than RAP's own run_pdm_score.py (RAP ships no v2 evaluation, so
# the paper's EPDMS came from the official devkit), and why conda activate rap is
# required (RAP's BEVFormer needs mmcv, which the navsim env cannot host).
#
# WHAT THE RL NUMBER DOES AND DOES NOT SAY
# ----------------------------------------
# The RL reward is v1 PDMS (navsim/agents/rap_dino/score_module/compute_navsim_score).
# navhard_two_stage reports EPDMS, which adds sub-scores the reward never saw
# (two-frame extended comfort, lane keeping) and a second stage of synthetic scenes.
# Improving the training reward therefore does not automatically improve this metric,
# and that gap is a result worth reporting, not a bug to tune away. Run the baseline
# column (RUN_BASELINE=1) so the comparison is like-for-like on the same split.

eval "$(conda shell.bash hook)"
conda activate rap

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"

NAVSIM_ROOT="$HOME/navsim"
RAP_ROOT="$HOME/RAP"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$RAP_ROOT/exp"
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# ORDER IS LOAD-BEARING. Both directories contain a package called "navsim", and both
# conda envs install one too. The devkit must win -- RAP has no navhard_two_stage
# config and the run would die on the first override. RAP_ROOT comes second only to
# make the `rl` package importable, which the devkit does not have. Reversing these
# two entries silently evaluates a different codebase.
export PYTHONPATH="$NAVSIM_ROOT:$RAP_ROOT"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# The agent config has to sit in the devkit's config package for hydra to find it by
# name; the copy in RAP is the source of truth. Same arrangement as rap_agent.yaml,
# which was copied in the same way. Refreshed on every run so an edit in the RAP repo
# cannot silently fail to take effect.
AGENT_CFG_DIR="$NAVSIM_ROOT/navsim/planning/script/config/common/agent"
cp "$RAP_ROOT/navsim/planning/script/config/common/agent/rl_agent.yaml" "$AGENT_CFG_DIR/rl_agent.yaml"

cd $NAVSIM_ROOT

TRAIN_TEST_SPLIT=navhard_two_stage

# The RAP weights the policy was trained against. This MUST be the checkpoint
# rl/precompute.py encoded the cache with -- the residual is defined relative to the
# trajectory that specific model emits, so a different checkpoint changes the base under
# the policy's feet without any error.
#
# Hence weights/RAP_DINO_navsimv2.ckpt (RLConfig.checkpoint_path), NOT the quicktest
# checkpoint that scripts/evaluation/run_navhard_evaluation.sh uses. Those are different
# models. Override both this and RLConfig.checkpoint_path together, or not at all.
CHECKPOINT=${CHECKPOINT:-$RAP_ROOT/weights/RAP_DINO_navsimv2.ckpt}

# PPO policy from rl/train.py. vecnormalize.pkl is picked up from the same directory.
POLICY=${POLICY:-$RAP_ROOT/exp/rap_rl_ppo/final_model.zip}

METRIC_CACHE_PATH=$HOME/navsim/metric_cache_navhard
NAVHARD_DATA_ROOT=$OPENSCENE_DATA_ROOT/navhard_two_stage

if [ ! -f "$POLICY" ]; then
    echo "No policy at $POLICY -- run scripts/rl/run_rl_train.sh first."
    exit 1
fi
if [ ! -f "$(dirname "$POLICY")/vecnormalize.pkl" ]; then
    echo "No vecnormalize.pkl next to $POLICY. It is part of the trained model;"
    echo "evaluating without it feeds the policy observations it never saw."
    exit 1
fi

run_eval () {
    # $1 experiment_name, $2 disable_residual
    python $NAVSIM_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=rl_agent \
        agent.checkpoint_path=$CHECKPOINT \
        agent.policy_path=$POLICY \
        agent.disable_residual=$2 \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.train_metric_cache_path=$METRIC_CACHE_PATH \
        worker=sequential \
        experiment_name=$1 \
        metric_cache_path=$METRIC_CACHE_PATH \
        synthetic_sensor_path=$NAVHARD_DATA_ROOT/sensor_blobs \
        synthetic_scenes_path=$NAVHARD_DATA_ROOT/synthetic_scene_pickles
}

# RUN_BASELINE=1 additionally scores the same weights with the residual forced to zero.
# That is plain RAP through the identical code path -- the only sound comparison, since
# it cancels any difference in feature building or proposal selection.
if [ "${RUN_BASELINE:-0}" = "1" ]; then
    echo "=== baseline: RAP, residual disabled ==="
    run_eval rl_navhard_baseline True
fi

echo "=== RL: RAP + learned residual ==="
run_eval rap_rl_navhard_eval False
