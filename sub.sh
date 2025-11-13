#!/bin/bash
#SBATCH -t 35:59:59
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=500g
#SBATCH --partition=long
#SBATCH -J ShearNet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adfield@wpi.edu
#SBATCH -o ShearNet_%j.out
#SBATCH -e ShearNet_%j.err

module load miniconda3
source activate shearnet_gpu

echo "Proceeding with code..."

shearnet-train --config configs/shearnet/forklike/superbit_psf/high_noise.yaml

shearnet-eval --model_name FIRST_const_sigma+flux_nse_sd_12_type_exp
