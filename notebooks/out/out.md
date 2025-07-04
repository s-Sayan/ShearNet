# ShearNet Notebook Output

Generated on: 2025-07-04 00:32:21

Output directory: `/home/adfield/ShearNet_Dev/notebooks/out`

---

==================================================

BENCHMARK CONFIGURATION

==================================================

Models to compare: ['Research ResNet', 'Research ResNet with PSF']

Include NGMix: False

==================================================


## Test Dataset Generation

Generated 5000 test samples

Image shape: (5000, 53, 53)

Labels shape: (5000, 4)

```
test_images stats: shape=(5000, 53, 53), min=-0.005, max=0.175, mean=0.001, std=0.005
```

```
test_labels stats: shape=(5000, 4), min=-0.949, max=5.000, mean=0.873, std=1.389
```

---


## Learning Curves Comparison

Research ResNet:

  Final training loss: 0.000006

  Final validation loss: 0.000011

  Best validation loss: 0.000011 at epoch 290

  Total epochs: 300

Research ResNet with PSF:

  Final training loss: 0.000003

  Final validation loss: 0.000009

  Best validation loss: 0.000009 at epoch 300

  Total epochs: 300

![learning_curves_comparison_20250704_003232.png](learning_curves_comparison_20250704_003232.png)

---


## Model Loading and Evaluation


Evaluating Research ResNet...


Evaluating Research ResNet with PSF...

---

