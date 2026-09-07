#!/bin/bash

#SBATCH --job-name=rap_train_metric_caching
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

## GAMMA training config
#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# OPTIONAL. Builds the navtrain PDM metric cache used by agent.config.pdm_scorer=True.
#
# DrivoR/exp/train_metric_cache is the same thing (navtrain, 103288 tokens, written by the
# identical train_metric_chache.MetricCache with pdm_progress), so run_rap_training.sh
# points at it by default and you do not need this. Only run it if you want an
# independent copy, e.g. one that keeps polygon interiors -- DrivoR's copy went through
# the exterior-ring compression in train_cache_processor.py:254.

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

cd $NAVSIM_DEVKIT_ROOT

CACHE_PATH=$NAVSIM_DEVKIT_ROOT/train_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py \
    train_test_split=navtrain \
    cache.cache_path=$CACHE_PATH \
    worker.threads_per_node=32
