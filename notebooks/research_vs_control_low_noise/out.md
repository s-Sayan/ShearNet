# ShearNet Notebook Output

Generated on: 2025-07-02 17:20:18

Output directory: `/home/adfield/ShearNet_Dev/notebooks/out`

---

==================================================

BENCHMARK CONFIGURATION

==================================================

Models to compare: ['Research ResNet', 'Control']

Include NGMix: False

==================================================


## Test Dataset Generation

Generated 5000 test samples

Image shape: (5000, 53, 53)

Labels shape: (5000, 4)

```
test_images stats: shape=(5000, 53, 53), min=-0.000, max=0.179, mean=0.001, std=0.005
```

```
test_labels stats: shape=(5000, 4), min=-0.949, max=5.000, mean=0.870, std=1.389
```

---


## Learning Curves Comparison

Research ResNet:

  Final training loss: 0.000006

  Final validation loss: 0.000011

  Best validation loss: 0.000011 at epoch 290

  Total epochs: 300

Control:

  Final training loss: 0.000059

  Final validation loss: 0.000058

  Best validation loss: 0.000041 at epoch 52

  Total epochs: 72

![learning_curves_comparison_20250702_172032.png](learning_curves_comparison_20250702_172032.png)

---


## Model Loading and Evaluation


Evaluating Research ResNet...


Evaluating Control...


All evaluations complete! Models: ['Research ResNet', 'Control']

---


## Model Evaluation Summary

============================================================


### EVALUATION SUMMARY

============================================================


Research ResNet:

  g1   : RMSE = 0.004394, Bias = 0.000001

  g2   : RMSE = 0.003583, Bias = -0.000035

  sigma: RMSE = 0.003634, Bias = -0.000098

  flux : RMSE = 0.004473, Bias = -0.000031


Control:

  g1   : RMSE = 0.008959, Bias = -0.000832

  g2   : RMSE = 0.009950, Bias = -0.004901

  sigma: RMSE = 0.009048, Bias = 0.000740

  flux : RMSE = 0.010140, Bias = -0.000013


Ready for plotting with 2 models

---


## Prediction Comparison Plots

![prediction_comparison_20250702_172119.png](prediction_comparison_20250702_172119.png)

---


## Residuals Comparison Plots

![residuals_comparison_20250702_172126.png](residuals_comparison_20250702_172126.png)

---


## Multi-model benchmark complete!

