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

# SHEARNET 

# forklike

# ideal

# high noise
shearnet-train --config configs/shearnet/forklike/ideal_psf/high_noise.yaml

shearnet-eval --model_name fork-like_ideal_high-noise --plot

# low noise
shearnet-train --config configs/shearnet/forklike/ideal_psf/low_noise.yaml

shearnet-eval --model_name fork-like_ideal_low-noise --plot

# superbit

# high noise
shearnet-train --config configs/shearnet/forklike/superbit_psf/high_noise.yaml

shearnet-eval --model_name fork-like_superbit_high-noise --plot

# low noise
shearnet-train --config configs/shearnet/forklike/superbit_psf/low_noise.yaml

shearnet-eval --model_name fork-like_superbit_low-noise --plot

# old cnn

# ideal

# high noise
shearnet-train --config configs/shearnet/old_cnn/ideal_psf/high_noise.yaml

shearnet-eval --model_name old-cnn_ideal_high-noise --plot

# low noise
shearnet-train --config configs/shearnet/old_cnn/ideal_psf/low_noise.yaml

shearnet-eval --model_name old-cnn_ideal_low-noise --plot

# superbit

# high noise
shearnet-train --config configs/shearnet/old_cnn/superbit_psf/high_noise.yaml

shearnet-eval --model_name old-cnn_superbit_high-noise --plot

# low noise
shearnet-train --config configs/shearnet/old_cnn/superbit_psf/low_noise.yaml

shearnet-eval --model_name old-cnn_superbit_low-noise --plot

# DECONVNET

# Research Backed

# ideal

# normalized

# high noise
deconvnet-train --config configs/deconvnet/research_backed/ideal_psf/normalized/high_noise.yaml

deconvnet-eval --model_name research_backed_ideal_normalized_high-noise --plot

# low noise
deconvnet-train --config configs/deconvnet/research_backed/ideal_psf/normalized/low_noise.yaml

deconvnet-eval --model_name research_backed_ideal_normalized_low-noise --plot

# not normalized

# high noise
deconvnet-train --config configs/deconvnet/research_backed/ideal_psf/not_normalized/high_noise.yaml

deconvnet-eval --model_name research_backed_ideal_not-normalized_high-noise --plot

# low noise
deconvnet-train --config configs/deconvnet/research_backed/ideal_psf/not_normalized/low_noise.yaml

deconvnet-eval --model_name research_backed_ideal_not-normalized_low-noise --plot

# superbit

# normalized

# high noise
deconvnet-train --config configs/deconvnet/research_backed/superbit_psf/normalized/high_noise.yaml

deconvnet-eval --model_name research_backed_superbit_normalized_high-noise --plot

# low noise
deconvnet-train --config configs/deconvnet/research_backed/superbit_psf/normalized/low_noise.yaml

deconvnet-eval --model_name research_backed_superbit_normalized_low-noise --plot

# not normalized

# high noise
deconvnet-train --config configs/deconvnet/research_backed/superbit_psf/not_normalized/high_noise.yaml

deconvnet-eval --model_name research_backed_superbit_not-normalized_high-noise --plot

# low noise
deconvnet-train --config configs/deconvnet/research_backed/superbit_psf/not_normalized/low_noise.yaml

deconvnet-eval --model_name research_backed_superbit_not-normalized_low-noise --plot
