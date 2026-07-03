#!/bin/bash

set -euo pipefail

# --- self-locate -------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
CONFIG="$ROOT/config.yaml"

# --- options -----------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [--no-train] [stage ...]

  --no-train        Skip the training job and submit the benchmarks with no
                    training dependency, reusing the existing model checkpoint.
                    Use this to re-run benchmarking after a benchmark job
                    failed, WITHOUT retraining the network.
                    (aliases: --skip-train, --benchmark-only)

  stage             One or more of: mcal leakage timing (default: all).
                    Selects which benchmark stages to submit.

Examples:
  $(basename "$0")                     # train, then run all benchmarks (default)
  $(basename "$0") --no-train          # re-run all benchmarks, no retraining
  $(basename "$0") --no-train leakage  # re-run only the leakage benchmark
EOF
}

SKIP_TRAIN=0
STAGES=()
for arg in "$@"; do
    case "$arg" in
        --no-train|--skip-train|--benchmark-only) SKIP_TRAIN=1 ;;
        -h|--help) usage; exit 0 ;;
        mcal|leakage|timing) STAGES+=("$arg") ;;
        *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
    esac
done
# default: submit every benchmark stage
if [[ ${#STAGES[@]} -eq 0 ]]; then
    STAGES=(mcal leakage timing)
fi

run_stage() {
    local want="$1" s
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$want" ]] && return 0
    done
    return 1
}

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
# DEP is passed UNQUOTED to submit_job: empty -> no dependency arg,
# non-empty -> a single --dependency=... arg (the string has no spaces).
DEP=""
AFTER=""
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    JOBID_TRAIN=$(submit_job train_sn training train_sn.sh)
    echo "Submitted training job:  $JOBID_TRAIN"
    DEP="--dependency=afterok:$JOBID_TRAIN"
    AFTER=" (after $JOBID_TRAIN)"
else
    echo "Skipping training — reusing existing model checkpoint."
fi

if run_stage mcal; then
    JOBID_M=$(submit_job mcal_ngmix_sn benchmarking/m run_mcal.sh $DEP)
    echo "Submitted bias job:      $JOBID_M$AFTER"
fi

if run_stage leakage; then
    JOBID_PSF=$(submit_job leakage benchmarking/psf_leakage run_leakage.sh $DEP)
    echo "Submitted leakage job:   $JOBID_PSF$AFTER"
fi

if run_stage timing; then
    JOBID_TIMING=$(submit_job timing timing run_timing.sh $DEP)
    echo "Submitted timing job:    $JOBID_TIMING$AFTER"
fi
