# #!/bin/bash

# #SBATCH --job-name=rap_visualize_raster
# #SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
# #SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

# ## No GPU/model needed -- this only queries the map API and rasterizes a scene
# #SBATCH --mem=32gb
# #SBATCH --ntasks=4

# #SBATCH --time=0:30:00
# #SBATCH --qos=huge-long
# #SBATCH --account=gamma
# #SBATCH --partition=gamma

# eval "$(conda shell.bash hook)"
# conda activate rap

# export HOME="/fs/nexus-projects/sim2real/aliu"
# export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
# export NAVSIM_DEVKIT_ROOT="$HOME/RAP"
# export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
# export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"

# # Change to RAP root so relative imports (process_data/helpers/renderer.py) resolve correctly
# cd $NAVSIM_DEVKIT_ROOT

# # --------------------------- UPDATE THESE ---------------------------
# TOKEN=""              # scene token to render; leave empty to use the first available one in the split
# CAMERA=cam_f0          # cam_f0/l0/l1/l2/r0/r1/r2/b0
# NAVSIM_LOG_PATH=$OPENSCENE_DATA_ROOT/navsim_logs/mini
# SENSOR_BLOBS_PATH=$OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini
# OUTPUT_DIR=$NAVSIM_DEVKIT_ROOT/raster_viz
# PERTURB=false          # true: also render a synthetic recovery viewpoint (xy+yaw jitter of the real ego pose)
# CROSS_AGENT=false      # true: also render a real cross-agent viewpoint (another real vehicle's logged pose becomes ego)
# # --------------------------- UPDATE THESE ---------------------------

# TOKEN_ARG=""
# if [ -n "$TOKEN" ]; then
#     TOKEN_ARG="--token $TOKEN"
# fi

# PERTURB_ARG=""
# if [ "$PERTURB" = "true" ]; then
#     PERTURB_ARG="--perturb"
# fi

# CROSS_AGENT_ARG=""
# if [ "$CROSS_AGENT" = "true" ]; then
#     CROSS_AGENT_ARG="--cross_agent"
# fi

# python $NAVSIM_DEVKIT_ROOT/scripts/evaluation/visualize_3d_rasterization.py \
#     --navsim_log_path $NAVSIM_LOG_PATH \
#     --sensor_blobs_path $SENSOR_BLOBS_PATH \
#     --output_dir $OUTPUT_DIR \
#     --camera $CAMERA \
#     $TOKEN_ARG \
#     $PERTURB_ARG \
#     $CROSS_AGENT_ARG


export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/RAP"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"
cd $NAVSIM_DEVKIT_ROOT

python scripts/evaluation/visualize_3d_rasterization.py \
    --navsim_log_path $OPENSCENE_DATA_ROOT/navsim_logs/mini \
    --sensor_blobs_path $OPENSCENE_DATA_ROOT/mini_sensor_blobs/mini \
    --output_dir ./raster_viz \
    --perturb \
    --cross_agent \
    --separate