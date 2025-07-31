# ShearNet Notebook Output

Generated on: 2025-07-30 18:43:50

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
test_galaxy_images stats: shape=(5000, 53, 53), min=-0.000, max=0.182, mean=0.001, std=0.005
```

```
test_psf_images stats: shape=(5000, 53, 53), min=-0.000, max=0.049, mean=0.000, std=0.003
```

```
test_labels stats: shape=(5000, 4), min=-0.949, max=5.000, mean=0.874, std=1.396
```

---


## Learning Curves Comparison

Fork-like:

  Final training loss: 0.000003

  Final validation loss: 0.000011

  Best validation loss: 0.000011 at epoch 300

  Total epochs: 300

Fork-like with High Samples:

  Final training loss: 0.000013

  Final validation loss: 0.000019

  Best validation loss: 0.000014 at epoch 156

  Total epochs: 176

![learning_curves_comparison_20250730_184407.png](learning_curves_comparison_20250730_184407.png)

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

  g1   : RMSE = 0.004867, Bias = -0.002383

  g2   : RMSE = 0.004740, Bias = 0.001130

  sigma: RMSE = 0.003350, Bias = -0.001517

  flux : RMSE = 0.005424, Bias = -0.002056


Fork-like with High Samples:

  g1   : RMSE = 0.004867, Bias = -0.002383

  g2   : RMSE = 0.004740, Bias = 0.001130

  sigma: RMSE = 0.003350, Bias = -0.001517

  flux : RMSE = 0.005424, Bias = -0.002056


Ready for plotting with 2 models

---


## Prediction Comparison Plots

![prediction_comparison_20250730_184438.png](prediction_comparison_20250730_184438.png)

---


## Residuals Comparison Plots

![residuals_comparison_20250730_184448.png](residuals_comparison_20250730_184448.png)

---


## Multi-model benchmark complete!

