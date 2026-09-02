#!/bin/bash

#SBATCH --job-name=rap_process_data
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

# No GPU: rendering is cv2 on numpy. Memory is per --thread-num worker (~8 GB each).
#SBATCH --mem=120gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=5:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

# Delete the nuPlan .db logs that no pipeline step reads any more.
# Both create_openscene_metadata_purturbed.py and create_openscene_metadata_aug.py
# filter to navtrain-minus-val, so only those 287 logs are ever opened.
# Regenerating these means re-downloading from nuscenes.org -- nothing else.
set -euo pipefail
LIST="/fs/nexus-projects/sim2real/aliu/RAP/process_data/db_delete_list.txt"  # absolute: sbatch copies this script to its spool dir, so $0 is unreliable
DB="/fs/nexus-projects/sim2real/aliu/nuplan/dataset/nuplan-v1.1/splits/trainval"
echo "before: $(ls "$DB" | wc -l) .db logs"
xargs -0 -a "$LIST" -P 16 -n 50 rm -f
echo "after:  $(ls "$DB" | wc -l) .db logs   (expected 287)"
df -h "$DB" | tail -1
