#!/bin/bash
#
# Submit one variation: train, then evaluate. One job, one output file.
#
#   ./sub.sh fourth_inloop_shearnet                    # train, then evaluate
#   ./sub.sh fourth_inloop_shearnet --no-train         # evaluate the checkpoint
#   ./sub.sh fourth_inloop_shearnet --baseline both    # ngmix AND the AnaCal fit
#
# ShearNet is always evaluated; --baseline picks what against. Every estimator
# is measured under every correction it admits, and the whole table always goes
# to one FITS -- there is no stage selection and no partial output, because a
# benchmark that sometimes writes half its columns produces results that cannot
# be compared to each other.
#
#   --baseline ngmix   esheldon/ngmix (default). metacal + sim.
#   --baseline anacal  AnaCal's Gaussian model fit, which carries an ANALYTIC
#                      shear response. Its fit is serial, so it is the
#                      expensive one: ~7 h at n_obs 200000 against ngmix's ~1.2.
#   --baseline both    the pair.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"

usage() {
    cat <<USAGE
Usage: $(basename "$0") <variation> [--no-train] [--estimators LIST]

  <variation>        directory name under research/unit_test_variations/
  --no-train         skip training; evaluate the existing checkpoint
  --baseline WHICH   ngmix (default), anacal, or both

Variations with an in-loop config:
$(cd "$ROOT" && grep -ls 'generation:[[:space:]]*inloop' */config.yaml 2>/dev/null \
    | xargs -r -n1 dirname | sort | sed 's/^/  /')
USAGE
}

[[ $# -ge 1 ]] || { usage; exit 1; }
case "$1" in -h|--help) usage; exit 0 ;; esac

VARIATION="$1"; shift
CONFIG="$ROOT/$VARIATION/config.yaml"
[[ -f "$CONFIG" ]] || { echo "No config at $CONFIG" >&2; usage; exit 1; }

SKIP_TRAIN=0
BASELINE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-train|--skip-train|--benchmark-only) SKIP_TRAIN=1; shift ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --baseline=*) BASELINE="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

LOGDIR="$ROOT/$VARIATION/logs"
mkdir -p "$LOGDIR"

echo "Variation:  $VARIATION"
echo "Config:     $CONFIG"
echo "Baseline:   ${BASELINE:-<from config>}  (ShearNet is always evaluated)"
[[ "$SKIP_TRAIN" -eq 1 ]] && echo "Training:   skipped, reusing the checkpoint"

JOBID=$(sbatch --parsable \
    --job-name="$VARIATION" \
    --output="$LOGDIR/%j.out" \
    --error="$LOGDIR/%j.err" \
    --export="ALL,CONFIG=$CONFIG,REPO=$REPO,SKIP_TRAIN=$SKIP_TRAIN,BASELINE=$BASELINE" \
    <<'SBATCH'
#!/bin/bash
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=200G

echo "===================================="
echo "SLURM JOB STARTED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Config: $CONFIG"
echo "Start:  $(date)"
echo "===================================="

source "$REPO/setup_env.sh"
start_time=$(date +%s)

if [[ "${SKIP_TRAIN:-0}" -eq 0 ]]; then
    echo "--- training ---"
    shearnet-train --config "$CONFIG"
else
    echo "--- training skipped ---"
fi

echo "--- evaluating (ShearNet vs ${BASELINE:-config baseline}) ---"
if [[ -n "${BASELINE:-}" ]]; then
    python "$REPO/research/shear_bias/run.py" -c "$CONFIG" --baseline "$BASELINE"
else
    python "$REPO/research/shear_bias/run.py" -c "$CONFIG"
fi

end_time=$(date +%s)
runtime=$((end_time - start_time))
printf -v h "%02d" $((runtime/3600)); printf -v m "%02d" $(((runtime%3600)/60))
printf -v s "%02d" $((runtime%60))
echo "===================================="
echo "Finished: $(date)"
echo "Runtime:  ${h}:${m}:${s} (HH:MM:SS)"
echo "===================================="
SBATCH
)

echo "Submitted:  $JOBID"
echo "Logs:       $LOGDIR/${JOBID}.out"
