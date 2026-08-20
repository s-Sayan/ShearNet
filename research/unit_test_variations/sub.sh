#!/bin/bash
#
# Submit a unit-test VARIATION through the training-matched benchmark harness.
#
# Variations are config-only directories; this supplies the jobs they lack.
# Every stage is driven entirely by $CONFIG, so one copy serves all of them.
#
#   ./sub.sh fourth_inloop_response              # train, then all benchmarks
#   ./sub.sh fourth_inloop_control --no-train    # re-benchmark an existing model
#   ./sub.sh fourth_inloop_forklens m            # just the bias table
#
# Benchmarks run research/shear_bias/run.py, which measures ShearNet, ngmix and
# FPFS in one job under a single response protocol -- so the m values in the
# output are directly comparable, with ngmix's metacal and FPFS's analytic
# response reported alongside as checks on that protocol.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
STAGES_DIR="$ROOT/_stages"

usage() {
    cat <<USAGE
Usage: $(basename "$0") <variation> [--no-train] [stage ...]

  <variation>   directory name under research/unit_test_variations/
  --no-train    skip training; benchmark the existing checkpoint
  stage         one or more of: m leakage timing (default: all)

Available variations with an in-loop config:
$(cd "$ROOT" && ls -d */ 2>/dev/null | tr -d '/' | grep inloop | sed 's/^/  /')
USAGE
}

[[ $# -ge 1 ]] || { usage; exit 1; }
case "$1" in -h|--help) usage; exit 0 ;; esac

VARIATION="$1"; shift
CONFIG="$ROOT/$VARIATION/config.yaml"
[[ -f "$CONFIG" ]] || { echo "No config at $CONFIG" >&2; usage; exit 1; }

SKIP_TRAIN=0
STAGES=()
for arg in "$@"; do
    case "$arg" in
        --no-train|--skip-train|--benchmark-only) SKIP_TRAIN=1 ;;
        m|leakage|timing) STAGES+=("$arg") ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
    esac
done
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(m leakage timing)

run_stage() {
    local want="$1" s
    for s in "${STAGES[@]}"; do [[ "$s" == "$want" ]] && return 0; done
    return 1
}

# submit_job <jobname> <log-subdir> <stage-script> [extra sbatch args...]
submit_job() {
    local name="$1" logsub="$2" script="$3"; shift 3
    local logdir="$ROOT/$VARIATION/$logsub/logs"
    mkdir -p "$logdir"
    sbatch --parsable \
        --job-name="${VARIATION}_${name}" \
        --output="$logdir/${name}_%j.out" \
        --error="$logdir/${name}_%j.err" \
        --export="ALL,ROOT=$ROOT/$VARIATION,REPO=$REPO,CONFIG=$CONFIG" \
        "$@" \
        "$STAGES_DIR/$script"
}

echo "Variation: $VARIATION"
echo "Config:    $CONFIG"

DEP=""; AFTER=""
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    JOBID_TRAIN=$(submit_job train training run_train.sh)
    echo "Submitted training job:  $JOBID_TRAIN"
    DEP="--dependency=afterok:$JOBID_TRAIN"
    AFTER=" (after $JOBID_TRAIN)"
else
    echo "Skipping training — reusing the existing checkpoint."
fi

run_stage m       && echo "Submitted bias job:      $(submit_job m       benchmarking/m            run_m.sh       $DEP)$AFTER"
run_stage leakage && echo "Submitted leakage job:   $(submit_job leakage benchmarking/psf_leakage  run_leakage.sh $DEP)$AFTER"
run_stage timing  && echo "Submitted timing job:    $(submit_job timing  timing                    run_timing.sh  $DEP)$AFTER"
