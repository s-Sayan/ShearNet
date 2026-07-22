#!/bin/bash
# =============================================================================
# ShearNet hyperparameter sweep -- SLURM job array.
#
# Each array task trains ONE combo (a full run, e.g. 300k samples), so the sweep
# runs in parallel instead of serially for days. After the array finishes, a
# tiny collect step ranks everything.
#
# ---- sizing the array -------------------------------------------------------
# The number of combos depends on your sweep YAML. Get it first, then set
# --array accordingly:
#
#     N=$(python research/hyperparam_search/sweep.py \
#            --sweep research/hyperparam_search/example_sweep.yaml --count)
#     sbatch --array=0-$((N-1))%8 research/hyperparam_search/sweep.sbatch
#
# The example_sweep.yaml grid is 3*3*2*2 = 36 combos -> --array=0-35.
# The "%8" caps concurrent tasks at 8 (tune to your GPU quota).
#
# ---- collect (after the array completes) ------------------------------------
#     python research/hyperparam_search/sweep.py \
#         --sweep research/hyperparam_search/example_sweep.yaml --collect
# or submit it as a dependent job (see the commented sbatch at the bottom).
#
# ---- quick preview (NO compute, run on the login node, no sbatch needed) -----
#     python research/hyperparam_search/sweep.py \
#         --sweep research/hyperparam_search/example_sweep.yaml --dry-run
# To dry-run the array itself: sbatch --export=ALL,EXTRA_ARGS=--dry-run ...
# =============================================================================

#SBATCH --job-name=shearnet-sweep
#SBATCH --array=0-35%8            # <-- set 0-(N-1) from `sweep.py --count`; %8 throttles concurrency
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00           # <-- per-combo walltime; raise for large `dataset.samples`
#SBATCH --output=slurm_logs/sweep_%A_%a.out
#SBATCH --error=slurm_logs/sweep_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

# --- environment (EDIT for your cluster) -------------------------------------
# module load cuda/12                       # if your site uses modules
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate shearnet_gpu               # the GPU env from `make install-gpu`
# export SHEARNET_DATA_PATH=/path/to/your/SHEARNET_DATA   # where models/plots are written
# -----------------------------------------------------------------------------

SWEEP="${SWEEP:-research/hyperparam_search/example_sweep.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"     # e.g. sbatch --export=ALL,EXTRA_ARGS=--dry-run ...

echo "[sweep.sbatch] job=${SLURM_ARRAY_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} sweep=${SWEEP}"

srun python research/hyperparam_search/sweep.py \
    --sweep "${SWEEP}" \
    --index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA_ARGS}

# -----------------------------------------------------------------------------
# Optional: submit the collect step as a job that waits for the whole array:
#
#   AID=$(sbatch --parsable --array=0-$((N-1))%8 research/hyperparam_search/sweep.sbatch)
#   sbatch --dependency=afterany:${AID} --wrap \
#     "python research/hyperparam_search/sweep.py --sweep ${SWEEP} --collect"
# -----------------------------------------------------------------------------
