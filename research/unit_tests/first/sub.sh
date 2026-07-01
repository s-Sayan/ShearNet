#!/bin/bash

set -euo pipefail

# --- self-locate -------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
CONFIG="$ROOT/config.yaml"

# --- submit helper -----------------------------------------------
# submit_job <jobname> <stage-subdir> <script> [extra sbatch args...]
submit_job() {
    local name="$1" stage="$2" script="$3"; shift 3
    local logdir="$ROOT/$stage/logs"
    mkdir -p "$logdir"
    sbatch --parsable \
        --job-name="$name" \
        --output="$logdir/${name}_%j.out" \
        --error="$logdir/${name}_%j.err" \
        --export="ALL,ROOT=$ROOT,REPO=$REPO,CONFIG=$CONFIG" \
        "$@" \
        "$ROOT/$stage/$script"
}

# --- pipeline ----------------------------------------------------
JOBID_TRAIN=$(submit_job train_sn training train_sn.sh)
echo "Submitted training job:  $JOBID_TRAIN"

JOBID_M=$(submit_job mcal_ngmix_sn benchmarking/m run_mcal.sh \
            --dependency=afterok:"$JOBID_TRAIN")
echo "Submitted bias job:      $JOBID_M (after $JOBID_TRAIN)"

JOBID_PSF=$(submit_job leakage benchmarking/psf_leakage run_leakage.sh \
            --dependency=afterok:"$JOBID_TRAIN")
echo "Submitted leakage job:   $JOBID_PSF (after $JOBID_TRAIN)"

JOBID_TIMING=$(submit_job timing timing run_timing.sh \
            --dependency=afterok:"$JOBID_TRAIN")
echo "Submitted timing job:    $JOBID_TIMING (after $JOBID_TRAIN)"
