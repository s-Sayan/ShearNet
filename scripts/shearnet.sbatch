#!/bin/bash
#SBATCH --job-name=shearnet
#SBATCH -p long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#
# The one Slurm script: it runs shearnet-train and shearnet-eval, nothing else.
# Submit it from the repository root.
#
#   sbatch scripts/shearnet.sbatch CONFIG [RUN_DIR]
#       train CONFIG (into RUN_DIR, else its run_options.outdir), then evaluate
#       the run with its own evaluation settings. Training failing stops the job.
#
#   sbatch scripts/shearnet.sbatch --eval-only RUN_DIR [EVAL_CONFIG EVAL_NAME]
#       evaluate a finished run again, optionally with evaluation overrides
#       under their own name.
#
#   sbatch scripts/shearnet.sbatch --train-only CONFIG [RUN_DIR]
#       train without evaluating (a hyperparameter sweep ranks on val loss).
#
#   sbatch --array=0-27%4 scripts/shearnet.sbatch --list configs/paper/runs.txt
#       one line of the list per array task (0-based, comments and blank lines
#       skipped). A line is the arguments of one of the forms above,
#       whitespace separated, so paths cannot contain spaces.
#
# The #SBATCH lines above are this cluster's defaults; override any of them on
# the sbatch command line (sbatch --time=12:00:00 ...).
#
# Environment: $SHEARNET_ENV (a script to source: conda activation, module
# loads) if set, else ./setup_env.sh if the checkout has one. Extra flags for
# the two commands go in $SHEARNET_TRAIN_ARGS / $SHEARNET_EVAL_ARGS (e.g.
# SHEARNET_TRAIN_ARGS=--overwrite to restart a failed run in place).

set -euo pipefail

usage() { sed -n '12,38p' "$0" | sed 's/^# \{0,1\}//'; }

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${SHEARNET_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$SHEARNET_ENV"
elif [[ -f setup_env.sh ]]; then
    # shellcheck disable=SC1091
    source setup_env.sh
fi
export MPLBACKEND=Agg
# float64 rendering for the in-loop trainer and the evaluation; must be set
# before Python starts, it cannot be changed inside a process
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

if [[ "${1:-}" == --list ]]; then
    LIST="${2:?--list needs a file}"
    INDEX="${SLURM_ARRAY_TASK_ID:?--list runs as a job array: sbatch --array=...}"
    LINE=$(grep -v '^[[:space:]]*\(#\|$\)' "$LIST" | sed -n "$((INDEX + 1))p")
    [[ -n "$LINE" ]] || { echo "no line $INDEX in $LIST" >&2; exit 1; }
    read -r -a FIELDS <<< "$LINE"
    set -- "${FIELDS[@]}"
fi

MODE=train
case "${1:-}" in
    -h|--help|"") usage; exit 1 ;;
    --eval-only) MODE=eval; shift ;;
    --train-only) MODE=train-only; shift ;;
esac

started=$(date +%s)
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

if [[ "$MODE" == train || "$MODE" == train-only ]]; then
    CONFIG="${1:?pass the config}"
    RUN_DIR="${2:-}"
    if [[ -z "$RUN_DIR" ]]; then
        RUN_DIR=$(python -c 'import sys; from shearnet.config import Config
print(Config.from_file(sys.argv[1]).get("run_options.outdir") or "")' "$CONFIG")
        [[ -n "$RUN_DIR" ]] || { echo "$CONFIG has no run_options.outdir; pass RUN_DIR" >&2; exit 1; }
    fi
    echo "config: $CONFIG"
    echo "run:    $RUN_DIR"
    # shellcheck disable=SC2086
    shearnet-train --config "$CONFIG" --run "$RUN_DIR" ${SHEARNET_TRAIN_ARGS:-}
    if [[ "$MODE" == train ]]; then
        # shellcheck disable=SC2086
        shearnet-eval --run "$RUN_DIR" ${SHEARNET_EVAL_ARGS:-}
    fi
else
    RUN_DIR="${1:?pass the run directory}"
    EVAL_ARGS=(--run "$RUN_DIR")
    if [[ -n "${2:-}" ]]; then
        EVAL_ARGS+=(--config "$2" --eval-name "${3:?an evaluation with overrides needs a name}")
    fi
    # shellcheck disable=SC2086
    shearnet-eval "${EVAL_ARGS[@]}" ${SHEARNET_EVAL_ARGS:-}
fi

elapsed=$(( $(date +%s) - started ))
printf 'done in %02d:%02d:%02d\n' $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
