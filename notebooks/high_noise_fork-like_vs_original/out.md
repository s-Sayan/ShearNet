# ShearNet Notebook Output

Generated on: 2025-08-16 16:51:26

Output directory: `/home/adfield/ShearNet/notebooks/out`

---

============================================================


## MODULAR BENCHMARK CONFIGURATION

============================================================

ShearNet models to compare: ['Fork-like - High Noise', 'Original - High Noise']

NGmix configs to compare: NO NGMIX

Total methods to evaluate: 2

============================================================


## Test Dataset Generation

Generated 5000 shared test samples for plotting

Galaxy image shape: (5000, 53, 53)

PSF image shape: (5000, 53, 53)

Labels shape: (5000, 4)

```
test_galaxy_images stats: shape=(5000, 53, 53), min=-0.000, max=0.177, mean=0.001, std=0.005
```

```
test_psf_images stats: shape=(5000, 53, 53), min=-0.000, max=0.049, mean=0.000, std=0.003
```

```
test_labels stats: shape=(5000, 4), min=-0.949, max=4.999, mean=0.866, std=1.383
```

---


## Learning Curves Comparison

Fork-like - High Noise training stats:

  Final training loss: 0.000096

  Final validation loss: 0.000138

  Best validation loss: 0.000138 at epoch 296

  Total epochs: 300

Original - High Noise training stats:

  Final training loss: 0.000158

  Final validation loss: 0.000249

  Best validation loss: 0.000198 at epoch 69

  Total epochs: 89

![learning_curves_comparison_20250816_165133.png](learning_curves_comparison_20250816_165133.png)

---


## Model Loading and Evaluation


### 
Evaluating Fork-like - High Noise...

Loading training config for Fork-like - High Noise: /home/adfield/ShearNet/plots/fork-like_high_noise/training_config.yaml

Generating test data for Fork-like - High Noise:

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

Found 1 matching directories for Fork-like - High Noise: ['fork-like_high_noise300']

Loading Fork-like - High Noise from: /home/adfield/ShearNet/model_checkpoint/fork-like_high_noise300

Model checkpoint loaded successfully.

Successfully evaluated Fork-like - High Noise

  MSE: 3.445e-04

  Bias: -4.058e-04


### 
Evaluating Original - High Noise...

Loading training config for Original - High Noise: /home/adfield/ShearNet/plots/original_high_noise/training_config.yaml

Generating test data for Original - High Noise:

  Samples: 5000

  PSF sigma: 0.25

  Noise SD: 0.001

  Experiment: ideal

  Stamp size: 53

  Pixel size: 0.141

  PSF shear: True

  Process PSF: False

Loading architecture from: /home/adfield/ShearNet/plots/original_high_noise/architecture.py

Model type: cnn

Successfully loaded model: EnhancedGalaxyNN

Found 1 matching directories for Original - High Noise: ['original_high_noise89']

Loading Original - High Noise from: /home/adfield/ShearNet/model_checkpoint/original_high_noise89

Model checkpoint loaded successfully.

Successfully evaluated Original - High Noise

  MSE: 4.242e-04

  Bias: +4.479e-03


### 
No NGmix configurations to evaluate.


All evaluations complete! Methods: ['Fork-like - High Noise', 'Original - High Noise']

---


## Model Evaluation Summary

============================================================


### EVALUATION SUMMARY

============================================================


Fork-like - High Noise (SHEARNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-03

    Experiment: ideal

    PSF Shear: True

  Performance:

    Overall MSE: 3.445e-04

    Overall Bias: -4.058e-04

    g1 MSE: 4.279e-04

    g2 MSE: 4.452e-04

    σ MSE: 6.116e-05

    Flux MSE: 4.437e-04

    Evaluation Time: 27.84 seconds


Original - High Noise (SHEARNET):

  Test Configuration:

    Samples: 5000

    PSF σ: 0.25

    Noise SD: 1.0e-03

    Experiment: ideal

    PSF Shear: True

  Performance:

    Overall MSE: 4.242e-04

    Overall Bias: +4.479e-03

    g1 MSE: 4.527e-04

    g2 MSE: 4.591e-04

    σ MSE: 1.153e-04

    Flux MSE: 6.699e-04

    Evaluation Time: 1.48 seconds


Ready for plotting with 2 methods

---


## Prediction Comparison Plots

![prediction_comparison_20250816_165233.png](prediction_comparison_20250816_165233.png)

---


## Residuals Comparison Plots

![residuals_comparison_20250816_165241.png](residuals_comparison_20250816_165241.png)

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

- all_results: ['Fork-like - High Noise', 'Original - High Noise']

- all_configs: ['Fork-like - High Noise', 'Original - High Noise']

- Individual test configurations for each method

- Both shared and fair evaluation results for comprehensive analysis


Successfully evaluated: 2 ShearNet models, 0 NGmix configs

---


## Modular benchmark complete!

