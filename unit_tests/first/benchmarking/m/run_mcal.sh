#!/bin/bash
#SBATCH -J mcal_ngmix_sn
#SBATCH --output=/home/adfield/ShearNet/unit_tests/first/benchmarking/m/logs/mcal_%j.out
#SBATCH --error=/home/adfield/ShearNet/unit_tests/first/benchmarking/m/logs/mcal_%j.err
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
# Avoid thread oversubscription
# ================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ================================
# Activate environment
# ================================
source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/etc/profile.d/conda.sh
conda activate shearnet_gpu

# ================================
# Run code and time execution
# ================================
start_time=$(date +%s)

python /home/adfield/ShearNet/shear_bias/m/main.py -c /home/adfield/ShearNet/unit_tests/first/benchmarking/m/config.yaml

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
