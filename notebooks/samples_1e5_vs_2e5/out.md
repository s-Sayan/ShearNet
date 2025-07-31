# ShearNet Notebook Output

Generated on: 2025-07-30 18:36:48

Output directory: `/home/adfield/ShearNet/notebooks/out`

---

==================================================

BENCHMARK CONFIGURATION

==================================================

Models to compare: ['Fork-like', 'Fork-like with High Samples']

Include NGMix: False

==================================================


## Test Dataset Generation

Generated 5000 test samples

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
test_labels stats: shape=(5000, 4), min=-0.949, max=5.000, mean=0.867, std=1.383
```

---


## Learning Curves Comparison

Fork-like:

  Final training loss: 0.000003

  Final validation loss: 0.000011

  Best validation loss: 0.000011 at epoch 300

  Total epochs: 300

Fork-like with High Samples:

  Final training loss: 0.000012

  Final validation loss: 0.000027

  Best validation loss: 0.000020 at epoch 77

  Total epochs: 97

![learning_curves_comparison_20250730_183733.png](learning_curves_comparison_20250730_183733.png)

---


## Model Loading and Evaluation


Evaluating Fork-like...


Evaluating Fork-like with High Samples...


All evaluations complete! Models: ['Fork-like', 'Fork-like with High Samples']

---


## Model Evaluation Summary

============================================================


### EVALUATION SUMMARY

============================================================


Fork-like:

  g1   : RMSE = 0.004895, Bias = -0.002368

  g2   : RMSE = 0.004839, Bias = 0.001071

  sigma: RMSE = 0.003665, Bias = -0.001521

  flux : RMSE = 0.005926, Bias = -0.002041


Fork-like with High Samples:

  g1   : RMSE = 0.004895, Bias = -0.002368

  g2   : RMSE = 0.004839, Bias = 0.001071

  sigma: RMSE = 0.003665, Bias = -0.001521

  flux : RMSE = 0.005926, Bias = -0.002041


Ready for plotting with 2 models

---


## Prediction Comparison Plots

![prediction_comparison_20250730_183801.png](prediction_comparison_20250730_183801.png)

---


## Residuals Comparison Plots

![residuals_comparison_20250730_183807.png](residuals_comparison_20250730_183807.png)

---


## Multi-model benchmark complete!

