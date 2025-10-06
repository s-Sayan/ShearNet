# ShearNet Notebook Output

Generated on: 2025-10-05 23:01:24

Output directory: `/home/adfield/ShearNet/notebooks/out`

---

============================================================


## DECONVNET COMPARISON CONFIGURATION

============================================================

DeconvNet models to compare: ['Research-Backed (Normalized, Low Noise)', 'Research-Backed (Not Normalized, Low Noise)']

GalSim configs to compare: NO GALSIM

Total methods to evaluate: 2

============================================================


## Test Dataset Generation

Generated 5000 shared test samples for plotting

Galaxy image shape: (5000, 53, 53)

PSF image shape: (5000, 53, 53)

Target image shape: (5000, 53, 53)

```
test_galaxy_images stats: shape=(5000, 53, 53), min=-0.000, max=0.174, mean=0.001, std=0.005
```

```
test_psf_images stats: shape=(5000, 53, 53), min=-0.000, max=0.049, mean=0.000, std=0.003
```

```
test_target_images stats: shape=(5000, 53, 53), min=-0.000, max=0.663, mean=0.001, std=0.007
```

---


## Learning Curves Comparison

Research-Backed (Normalized, Low Noise) training stats:

  Final training loss: 0.000420

  Final validation loss: 0.000905

  Best validation loss: 0.000217 at epoch 19

  Total epochs: 44

Research-Backed (Not Normalized, Low Noise) training stats:

  Final training loss: 0.000000

  Final validation loss: 0.000000

  Best validation loss: 0.000000 at epoch 19

  Total epochs: 44

![deconvnet_learning_curves_comparison_20251005_230135.png](deconvnet_learning_curves_comparison_20251005_230135.png)

---


## Model Loading and Evaluation


### 
Evaluating Research-Backed (Normalized, Low Noise)...

Loading training config for Research-Backed (Normalized, Low Noise): /home/adfield/ShearNet/plots/research_backed_ideal_normalized_low-noise/training_config.yaml

Generating test data for Research-Backed (Normalized, Low Noise):

  Samples: 5000

  PSF sigma: 0.25

  Noise SD: 1e-05

  Normalized: True

  Experiment: ideal

  Stamp size: 53

  Pixel size: 0.141

  PSF shear: True

Loading architecture from: /home/adfield/ShearNet/plots/research_backed_ideal_normalized_low-noise/architecture.py

Model type: research_backed

Successfully loaded model: ResearchBackedPSFDeconvolutionUNet

Found 1 matching directories for Research-Backed (Normalized, Low Noise): ['research_backed_ideal_normalized_low-noise44']

Loading Research-Backed (Normalized, Low Noise) from: /home/adfield/ShearNet/model_checkpoint/research_backed_ideal_normalized_low-noise44

Model checkpoint loaded successfully.

Successfully evaluated Research-Backed (Normalized, Low Noise)

  MSE: 1.555e-03

  PSNR: 67.01 dB

  SSIM: 0.9981


### 
Evaluating Research-Backed (Not Normalized, Low Noise)...

Loading training config for Research-Backed (Not Normalized, Low Noise): /home/adfield/ShearNet/plots/research_backed_ideal_not-normalized_low-noise/training_config.yaml

Generating test data for Research-Backed (Not Normalized, Low Noise):

  Samples: 5000

  PSF sigma: 0.25

  Noise SD: 1e-05

  Normalized: False

  Experiment: ideal

  Stamp size: 53

  Pixel size: 0.141

  PSF shear: True

Loading architecture from: /home/adfield/ShearNet/plots/research_backed_ideal_not-normalized_low-noise/architecture.py

Model type: research_backed

Successfully loaded model: ResearchBackedPSFDeconvolutionUNet

Found 1 matching directories for Research-Backed (Not Normalized, Low Noise): ['research_backed_ideal_not-normalized_low-noise44']

Loading Research-Backed (Not Normalized, Low Noise) from: /home/adfield/ShearNet/model_checkpoint/research_backed_ideal_not-normalized_low-noise44

Model checkpoint loaded successfully.

Successfully evaluated Research-Backed (Not Normalized, Low Noise)

  MSE: 5.662e-07

  PSNR: 58.77 dB

  SSIM: 0.9942


### 
No GalSim configurations to evaluate.


All evaluations complete! Methods: ['Research-Backed (Normalized, Low Noise)', 'Research-Backed (Not Normalized, Low Noise)']

---


## Model Evaluation Summary

============================================================


### EVALUATION SUMMARY

============================================================


Research-Backed (Normalized, Low Noise) (DECONVNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-05

    Experiment: ideal

    PSF Shear: True

  Performance:

    MSE: 1.555e-03

    MAE: 6.017e-03

    PSNR: 67.01 dB

    SSIM: 0.9981

    Bias: +2.751e-03

    Evaluation Time: 54.07 seconds


Research-Backed (Not Normalized, Low Noise) (DECONVNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-05

    Experiment: ideal

    PSF Shear: True

  Performance:

    MSE: 5.662e-07

    MAE: 1.710e-04

    PSNR: 58.77 dB

    SSIM: 0.9942

    Bias: -3.639e-05

    Evaluation Time: 42.86 seconds


Ready for plotting with 2 methods

---


## Deconvolution Comparison Plots

![deconvnet_comparison_20251005_230506.png](deconvnet_comparison_20251005_230506.png)

---


## Spatial Residuals Analysis

Creating spatial residuals comparison for 2 methods

![deconvnet_spatial_residuals_20251005_230509.png](deconvnet_spatial_residuals_20251005_230509.png)

---

