#!/bin/bash

#SBATCH --job-name=rap_warmup_eval
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

## Scale ntasks with gpus
#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=32

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

# Hugging Face token for gated repos (e.g. DINOv3). Create once with:
#   echo hf_xxxxxxx... > $HOME/.hf_token && chmod 600 $HOME/.hf_token
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# Change to RAP root so relative paths (e.g. checkpoint, cache dirs) resolve correctly
cd $NAVSIM_DEVKIT_ROOT

TRAIN_TEST_SPLIT=warmup_test_e2e
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/weights/RAP_DINO_navsimv2.ckpt
# Build this first with scripts/evaluation/run_metric_caching_warmup.sh
METRIC_CACHE_PATH=$NAVSIM_DEVKIT_ROOT/metric_cache_warmup

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=rap_agent \
    agent.checkpoint_path=$CHECKPOINT \
    agent.config.train_metric_cache_path=$METRIC_CACHE_PATH \
    agent.config.trajectory_sampling.time_horizon=5 \
    experiment_name=rap_warmup_eval \
    metric_cache_path=$METRIC_CACHE_PATH \
    navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/mini \
    sensor_blobs_path=$OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini
