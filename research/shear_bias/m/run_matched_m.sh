#!/bin/bash
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --mem=96G

# ================================
# Print job info
# ================================
echo "===================================="
echo "SLURM JOB STARTED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "===================================="

# ================================
# Activate environment
# ================================
: "${CONFIG:?CONFIG not set — launch through sub.sh}"
: "${REPO:?REPO not set — launch through sub.sh}"
source "$REPO/setup_env.sh"

# ================================
# Run the TRAINING-MATCHED harness: stamps are rendered by the backend
# recorded in the model's saved training_config.yaml, and every estimator
# is calibrated through the same renderer-response protocol. The legacy
# run_mcal.sh script beside this one still runs main.py; keeping both is how
# a simulator drift between them gets caught.
# ================================
start_time=$(date +%s)

python ../run.py --task m -c ../config.yaml

end_time=$(date +%s)
runtime=$((end_time - start_time))

# Format runtime
printf -v h "%02d" $((runtime/3600))
printf -v m "%02d" $(((runtime%3600)/60))
printf -v s "%02d" $((runtime%60))

echo "===================================="
echo "Job finished at: $(date)"
echo "Total runtime: ${h}:${m}:${s} (HH:MM:SS)"
echo "===================================="
