#!/bin/bash

#SBATCH --job-name=vis3d_odometry
#SBATCH --output=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j
#SBATCH --error=/fs/nexus-projects/sim2real/aliu/RAP/my_dump/%x.out.%j

#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --time=03:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma

set -euo pipefail

BASE=/fs/nexus-projects/sim2real/aliu/RAP
RCLONE=${RCLONE:-$HOME/bin/rclone}
SOURCE=${SOURCE:-$BASE/CARE}
DEST=${DEST:-"e:Autonomous Driving Research Dump/CARE"}
LOG=${LOG:-$HOME/care_upload.log}

[ -x "$RCLONE" ] || { echo "error: no rclone at $RCLONE" >&2; exit 1; }
[ -d "$SOURCE" ] || { echo "error: no such directory: $SOURCE" >&2; exit 1; }

FLAGS=(
    --transfers 16
    --drive-pacer-min-sleep 10ms    # Drive's API rate limiting: shorter waits between calls
    --drive-pacer-burst 200
    --stats 30s
    -v
)

if [ "${DRY_RUN:-0}" = 1 ]; then
    "$RCLONE" copy "$SOURCE" "$DEST" "${FLAGS[@]}" --dry-run
    exit 0
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Under sbatch: run in the foreground. Slurm ends a job when its script
    # exits and kills everything the job started, nohup or not -- a backgrounded
    # upload here dies within a second having transferred nothing. Progress goes
    # to the job's own output file (my_dump/<job name>.out.<job id>).
    echo "uploading $SOURCE -> $DEST (slurm job $SLURM_JOB_ID, foreground)"
    "$RCLONE" copy "$SOURCE" "$DEST" "${FLAGS[@]}"
    echo "upload finished"
else
    # From a terminal: detach, so closing the terminal does not stop it.
    nohup "$RCLONE" copy "$SOURCE" "$DEST" "${FLAGS[@]}" --log-file="$LOG" >/dev/null 2>&1 &
    echo "uploading $SOURCE -> $DEST (pid $!)"
    echo "progress: tail -f $LOG"
fi
