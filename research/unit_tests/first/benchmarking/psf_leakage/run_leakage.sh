#!/bin/bash
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
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
# Run code and time execution
# ================================
start_time=$(date +%s)

python "$REPO/research/shear_bias/psf_leakage/main.py" -c "$CONFIG"

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
