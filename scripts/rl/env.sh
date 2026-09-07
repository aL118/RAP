#!/bin/bash
# Shared environment for the RL scripts. Mirrors scripts/training/run_rap_training_quicktest.sh
# so the RL pipeline reads exactly the caches the supervised pipeline writes.
eval "$(conda shell.bash hook)"
conda activate rap

export HOME="/fs/nexus-projects/sim2real/aliu"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim/dataset/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/RAP"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export OPENSCENE_DATA_ROOT="$HOME/navsim/dataset"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Agent-independent PDM cache. The cache itself lives in DrivoR and is reused, not
# rebuilt (it is agent-independent -- see rl/config.py); RAP/exp/train_metric_cache is
# a symlink to it, added 2026-09-05 because both this variable and rl/config.py's
# default had been pointed at the RAP path while the data was still only in DrivoR.
# RLConfig.validate() catches that as a missing directory, so it fails fast rather
# than scoring every token as NaN -- but it does fail.
export RAP_METRIC_CACHE="$HOME/RAP/exp/train_metric_cache"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Scratch for tempfile. wandb and torch each build a TemporaryDirectory at *import*
# time with no dir= argument -- artifact.py:153 "wandb-artifacts", _private.py:6
# "wandb-media", distributed/nn/jit/instantiator.py:19 for _remote_module -- so all
# three land in tempfile.gettempdir(). That walks $TMPDIR, $TEMP, $TMP, /tmp, /var/tmp,
# /usr/tmp and falls back to the cwd, which for these scripts is the checkout. On
# gammagpu04 it reached that fallback and left tmp*wandb-artifacts, tmp*wandb-media and
# bare tmp* directories in the repo root (jobs 7435020, 7435526, 7435545). Why only that
# node is still uncharacterised -- the probes in job 7447541/7447582 landed on legacy02
# and gammagpu01, where /tmp is writable xfs and gettempdir() is /tmp, so the node that
# actually misbehaved was never measured. Pinning TMPDIR makes the answer irrelevant.
#
# Note this is TMPDIR's job, not WANDB_DIR's: WANDB_DIR moves the wandb/ run directory
# and has no effect on those two import-time temp dirs.
#
# Node-local /tmp is preferred when usable -- xfs, so no NFS silly-rename ".nfsXXXX
# Device or resource busy" noise on dataloader-worker teardown, and the node reclaims it.
# The project filesystem is the correct-but-NFS fallback.
_scratch=/tmp/rap_rl_${SLURM_JOB_ID:-manual$$}
if mkdir -p "$_scratch" 2>/dev/null && [ -w "$_scratch" ]; then
    export TMPDIR=$_scratch
else
    export TMPDIR=$NAVSIM_DEVKIT_ROOT/tmp/${SLURM_JOB_ID:-manual$$}
    mkdir -p "$TMPDIR"
    echo "note: /tmp unusable on $(hostname); scratch on NFS at $TMPDIR"
fi
echo "TMPDIR=$TMPDIR"
# Best effort: a SIGKILL (OOM, walltime, scancel) never runs this, which is exactly how
# the three jobs above orphaned theirs. The job id in the name keeps whatever survives
# attributable to the run that left it.
trap '[ -n "$TMPDIR" ] && rm -rf "$TMPDIR"' EXIT
