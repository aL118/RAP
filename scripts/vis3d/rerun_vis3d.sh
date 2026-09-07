#!/bin/bash
# Set the values below and run this directly (not sbatch -- it only submits).
# Everything else comes from run_render_vis3d.sh's own CONFIG block.
############################### CONFIG ###############################
DATASET="CARE_YTB"
VIDEO="closetruck"
RUN="1"

RUN_STAGE1_MASKS=0               # objects (GroundingDINO+SAM)
RUN_STAGE1B_LANES=0              # lane dividers (YOLOPv2)
RUN_STAGE2_DEPTH=0               # point maps (UniDepth)
RUN_STAGE3_LIFT=1                # lift boxes + rasterize overlays (CPU)
RUN_STAGE4_VO=0                  # ego trajectory (visual odometry)
EXPORT_NAVSIM=0                  # navsim-format log
######################################################################

set -euo pipefail

BASE=/fs/nexus-projects/sim2real/aliu/RAP

sbatch --job-name="vis3d_rerun_$VIDEO" \
    --export="ALL,DATASET=$DATASET,VIDEO=$VIDEO,RUN=$RUN,\
RUN_STAGE1_MASKS=$RUN_STAGE1_MASKS,RUN_STAGE1B_LANES=$RUN_STAGE1B_LANES,\
RUN_STAGE2_DEPTH=$RUN_STAGE2_DEPTH,RUN_STAGE3_LIFT=$RUN_STAGE3_LIFT,\
RUN_STAGE4_VO=$RUN_STAGE4_VO,EXPORT_NAVSIM=$EXPORT_NAVSIM" \
    "$BASE/scripts/vis3d/run_render_vis3d.sh"
