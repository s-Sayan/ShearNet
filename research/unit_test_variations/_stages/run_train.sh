#!/bin/bash
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --mem=200G

# Generic train stage for research/unit_test_variations/<name>/config.yaml.
# Launch through sub.sh, which sets CONFIG and REPO.
echo "===================================="
echo "SLURM JOB STARTED  (train)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo "===================================="

: "${CONFIG:?CONFIG not set — launch through sub.sh}"
: "${REPO:?REPO not set — launch through sub.sh}"
source "$REPO/setup_env.sh"

start_time=$(date +%s)
shearnet-train --config "$CONFIG"
end_time=$(date +%s)
runtime=$((end_time - start_time))
printf -v h "%02d" $((runtime/3600)); printf -v m "%02d" $(((runtime%3600)/60))
printf -v s "%02d" $((runtime%60))
echo "===================================="
echo "Job finished at: $(date)"
echo "Total runtime: ${h}:${m}:${s} (HH:MM:SS)"
echo "===================================="
