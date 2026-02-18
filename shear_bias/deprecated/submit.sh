#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem=180G
#SBATCH -p short
#SBATCH -J shear_bias
#SBATCH -v
#SBATCH -o logs/sbout.log
#SBATCH -e logs/sberr.log

echo "Submitted job with ID: $SLURM_JOB_ID"
# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"

source ~/.bashrc
conda activate shearnet

python ngmix_shear_bias.py -c config.yaml

# Record end time
end=$(date +%s)
echo "Job finished at: $(date)"

# Compute elapsed time
runtime=$((end - start_time))

# Optionally, print in minutes/hours
echo "Total runtime: $(awk -v t=$runtime 'BEGIN {
    h = int(t/3600);
    m = int((t%3600)/60);
    s = int(t%60);
    printf "%02d:%02d:%02d\n", h, m, s
}')"
