#!/bin/bash
#SBATCH --job-name=shearnet-submit
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Submit one config's job graph: training and ShearNet evaluation on GPUs, ngmix
# evaluation on CPUs, then the merge. Run it from the repository root, either
#
#   bash scripts/shearnet.sbatch CONFIG --time 4-00:00:00   # login node, no queue wait
#   sbatch --time=4-00:00:00 scripts/shearnet.sbatch CONFIG  # as a tiny job
#
# The time limit applies to each compute job. Under sbatch it is this job's own
# --time; with bash, pass --time (default 4-00:00:00). SHEARNET_TRAIN_ARGS=--overwrite
# starts a new generation next to the existing ones.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${SHEARNET_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$SHEARNET_ENV"
fi
# submitting needs no GPU; the stage jobs choose their own devices
export JAX_PLATFORMS=cpu
exec python -m shearnet.cli.pipeline submit "$@"
