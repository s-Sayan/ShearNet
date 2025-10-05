# ShearNet Notebook Output

Generated on: 2025-08-16 15:56:30

Output directory: `/home/adfield/ShearNet/notebooks/out`

---

============================================================


## MODULAR BENCHMARK CONFIGURATION

============================================================

ShearNet models to compare: ['ShearNet - High Noise', 'ShearNet - Low Noise']

NGmix configs to compare: NO NGMIX

Total methods to evaluate: 2

============================================================


## Test Dataset Generation

Generated 5000 shared test samples for plotting

Galaxy image shape: (5000, 53, 53)

PSF image shape: (5000, 53, 53)

Labels shape: (5000, 4)

```
test_galaxy_images stats: shape=(5000, 53, 53), min=-0.000, max=0.181, mean=0.001, std=0.005
```

```
test_psf_images stats: shape=(5000, 53, 53), min=-0.000, max=0.049, mean=0.000, std=0.003
```

```
test_labels stats: shape=(5000, 4), min=-0.949, max=4.999, mean=0.871, std=1.392
```

---


## Learning Curves Comparison

ShearNet - High Noise training stats:

  Final training loss: 0.000096

  Final validation loss: 0.000138

  Best validation loss: 0.000138 at epoch 296

  Total epochs: 300

ShearNet - Low Noise training stats:

  Final training loss: 0.000003

  Final validation loss: 0.000020

  Best validation loss: 0.000020 at epoch 278

  Total epochs: 300

![learning_curves_comparison_20250816_155638.png](learning_curves_comparison_20250816_155638.png)

---


## Model Loading and Evaluation


### 
Evaluating ShearNet - High Noise...

Loading training config for ShearNet - High Noise: /home/adfield/ShearNet/plots/fork-like_high_noise/training_config.yaml

Generating test data for ShearNet - High Noise:

  Samples: 5000

  PSF sigma: 0.25

  Noise SD: 0.001

  Experiment: ideal

  Stamp size: 53

  Pixel size: 0.141

  PSF shear: True

  Process PSF: True

Loading architecture from: /home/adfield/ShearNet/plots/fork-like_high_noise/architecture.py

Model type: fork-like

Galaxy type: research_backed, PSF type: forklens_psf

Successfully loaded model: ForkLike

Found 1 matching directories for ShearNet - High Noise: ['fork-like_high_noise300']

Loading ShearNet - High Noise from: /home/adfield/ShearNet/model_checkpoint/fork-like_high_noise300

Model checkpoint loaded successfully.

Successfully evaluated ShearNet - High Noise

  MSE: 3.505e-04

  Bias: -2.636e-04


### 
Evaluating ShearNet - Low Noise...

Loading training config for ShearNet - Low Noise: /home/adfield/ShearNet/plots/fork-like_low_noise/training_config.yaml

Generating test data for ShearNet - Low Noise:

  Samples: 5000

  PSF sigma: 0.25

  Noise SD: 1e-05

  Experiment: ideal

  Stamp size: 53

  Pixel size: 0.141

  PSF shear: True

  Process PSF: True

Loading architecture from: /home/adfield/ShearNet/plots/fork-like_low_noise/architecture.py

Model type: fork-like

Galaxy type: research_backed, PSF type: forklens_psf

Successfully loaded model: ForkLike

Found 1 matching directories for ShearNet - Low Noise: ['fork-like_low_noise300']

Loading ShearNet - Low Noise from: /home/adfield/ShearNet/model_checkpoint/fork-like_low_noise300

Model checkpoint loaded successfully.

Successfully evaluated ShearNet - Low Noise

  MSE: 2.094e-04

  Bias: -1.327e-04


### 
No NGmix configurations to evaluate.


All evaluations complete! Methods: ['ShearNet - High Noise', 'ShearNet - Low Noise']

---


## Model Evaluation Summary

============================================================


### EVALUATION SUMMARY

============================================================


ShearNet - High Noise (SHEARNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-03

    Experiment: ideal

    PSF Shear: True

  Performance:

    Overall MSE: 3.505e-04

    Overall Bias: -2.636e-04

    g1 MSE: 4.328e-04

    g2 MSE: 4.499e-04

    σ MSE: 5.986e-05

    Flux MSE: 4.593e-04

    Evaluation Time: 28.36 seconds


ShearNet - Low Noise (SHEARNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-05

    Experiment: ideal

    PSF Shear: True

  Performance:

    Overall MSE: 2.094e-04

    Overall Bias: -1.327e-04

    g1 MSE: 3.994e-04

    g2 MSE: 4.078e-04

    σ MSE: 2.506e-05

    Flux MSE: 5.516e-06

    Evaluation Time: 11.95 seconds


Ready for plotting with 2 methods

---


## Prediction Comparison Plots

![prediction_comparison_20250816_155747.png](prediction_comparison_20250816_155747.png)

---


## Residuals Comparison Plots

![residuals_comparison_20250816_155756.png](residuals_comparison_20250816_155756.png)

---


## Future Analysis Section

============================================================


### FUTURE ANALYSIS CAPABILITIES

============================================================

This section is reserved for future additional analysis and plots.

You can add new analysis here without changing the existing plots above.



Potential future features to add:

- Configuration impact analysis (noise level vs performance)

- Method comparison tables with statistical significance tests

- Performance vs computational cost scatter plots

- Bias vs noise level correlation analysis

- PSF shear impact visualization

- Galaxy type performance comparison

- Training configuration clustering analysis



To add new analysis, simply add code in this section.

The modular structure provides access to:

- all_results: ['ShearNet - High Noise', 'ShearNet - Low Noise']

- all_configs: ['ShearNet - High Noise', 'ShearNet - Low Noise']

- Individual test configurations for each method

- Both shared and fair evaluation results for comprehensive analysis


Successfully evaluated: 2 ShearNet models, 0 NGmix configs

---


## Modular benchmark complete!

