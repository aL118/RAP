#!/bin/bash

#SBATCH --job-name=rap_navhard_eval
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

# NAVSIM v2 navhard two-stage evaluation of RAP-DINO -> EPDMS (the paper's 39.6 metric).
#
# WHY THIS RUNS THE DEVKIT AND NOT RAP'S OWN run_pdm_score.py
# -----------------------------------------------------------
# RAP does not ship v2 evaluation. Checked against origin/main (5fd8630): zero files match
# navhard_two_stage, traffic_agents_policies, dataloader_navhard, pdm_score_after_fix,
# synthetic_scenes_path, TWO_FRAME_EXTENDED_COMFORT or LANE_KEEPING. RAP's run_pdm_score.py
# has 0 two-stage references (the devkit's has 68) and its pdm_scorer.py is v1 PDMS only --
# that is the README's 93.8 row, a DIFFERENT metric from the 39.6 EPDMS row.
#
# This is a known gap upstream, not a broken checkout: vita-epfl/RAP issue #17 asks
# "I do not see the EPDMS score. Do I need to run the eval on the navsim original codebase
# to get this?" -- the maintainer never answered and the issue is still open. Issue #13
# ("Error in metric caching") likewise has an unanswered "NAVSIM的代码跑不通". The only
# reproduction recipe the maintainers ever gave (issue #10) is train_test_split=navtest,
# i.e. v1 PDMS.
#
# So the paper's 39.6 was produced with the official navsim devkit. That is what this runs.
# RAP's agent (navsim/agents/rap_dino/) was copied into the devkit; RAP itself is untouched.
#
# THE TWO LINES THAT ARE EASY TO GET WRONG
# -----------------------------------------
# 1. conda activate rap, NOT navsim. RAP's BEVFormer imports mmengine/mmcv
#    (rap_dino/bevformer/custom_base_transformer_layer.py:6). The navsim env has neither and
#    cannot easily get them: it runs torch 2.8, while mmcv 2.1.0 is built against torch 2.1.
#    The rap env already has nuplan-devkit 1.2.0, ray, hydra-core 1.2.0 and identical
#    numpy/opencv/shapely, so the devkit's evaluation stack imports there cleanly.
#
# 2. PYTHONPATH=$NAVSIM_ROOT is REQUIRED. Both envs install a package called "navsim" (rap's
#    egg-link -> RAP, navsim's easy-install.pth -> the devkit). Running "python <path>/x.py"
#    puts the SCRIPT's directory on sys.path, never the cwd, so `cd` alone cannot decide
#    which one wins. In the rap env "import navsim" would otherwise resolve to RAP -- which
#    has no navhard_two_stage config -- and the run dies on the first override. PYTHONPATH
#    precedes site-packages, so this is what puts the devkit first.

eval "$(conda shell.bash hook)"
conda activate rap

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"

# The devkit supplies the evaluation code; results still land under RAP/exp.
NAVSIM_ROOT="$HOME/navsim"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$HOME/RAP/exp"
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export PYTHONPATH="$NAVSIM_ROOT"

# DINOv3 is a gated repo; RAPModel pulls the backbone at init.
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

cd $NAVSIM_ROOT

TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=/fs/nexus-projects/sim2real/aliu/RAP/exp/rap_navsim_train/2026.09.03.21.23.11/epoch9-step2020.ckpt
METRIC_CACHE_PATH=$HOME/navsim/metric_cache_navhard
NAVHARD_DATA_ROOT=$OPENSCENE_DATA_ROOT/navhard_two_stage

python $NAVSIM_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=rap_agent \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.train_metric_cache_path=$METRIC_CACHE_PATH \
    worker=sequential \
    experiment_name=rap_navhard_eval \
    metric_cache_path=$METRIC_CACHE_PATH \
    synthetic_sensor_path=$NAVHARD_DATA_ROOT/sensor_blobs \
    synthetic_scenes_path=$NAVHARD_DATA_ROOT/synthetic_scene_pickles
