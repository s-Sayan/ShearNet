This branch currently houses the first working deconvolution branch, below is the result of after evaluating 5000 galaxies!

Here are the results of my new research backed model:

```
=== Neural Deconvolution Results ===
Mean Squared Error (MSE): 3.167216e-07
Mean Absolute Error (MAE): 1.015540e-04
Peak Signal-to-Noise Ratio (PSNR): 61.42 dB
Structural Similarity Index (SSIM): 0.9997
Approximate Perceptual Distance: 5.491839e-07
Bias: -2.991935e-05
Normalized MSE: 5.772880e-03
Evaluation time: 37.45 seconds

=== NGmix Exponential Deconvolution ===
Evaluation Time: 12.24 seconds
Mean Squared Error (MSE): 2.093797e-05
Mean Absolute Error (MAE): 9.358793e-04
Peak Signal-to-Noise Ratio (PSNR): 43.21 dB
Bias: -2.517923e-05

=== NGmix Gaussian Deconvolution ===
Evaluation Time: 8.79 seconds
Mean Squared Error (MSE): 1.202073e-05
Mean Absolute Error (MAE): 4.009046e-04
Peak Signal-to-Noise Ratio (PSNR): 45.62 dB
Bias: -8.327550e-06

=== NGmix de Vaucouleurs Deconvolution ===
Evaluation Time: 255.30 seconds
Mean Squared Error (MSE): 4.257531e-05
Mean Absolute Error (MAE): 1.089797e-03
Peak Signal-to-Noise Ratio (PSNR): 40.13 dB
Bias: -7.164997e-04
```

See the config at [this directory](./configs/research_backed_deconv.yaml) to get more details of this task.