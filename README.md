This branch currently houses the first working deconvolution branch, below is the result of after evaluating 5000 galaxies!

```
=== Neural Deconvolution Results ===
Mean Squared Error (MSE): 2.670307e-07
Mean Absolute Error (MAE): 6.341388e-05
Peak Signal-to-Noise Ratio (PSNR): 62.10 dB
Structural Similarity Index (SSIM): 0.9997
Approximate Perceptual Distance: 6.029722e-07
Bias: -4.286317e-06
Normalized MSE: 4.841983e-03
Evaluation time: 23.58 seconds

Generating evaluation plots...

=== FFT Deconvolution Results ===
Evaluation Time: 1.31 seconds
Mean Squared Error (MSE): 2.034137e-06
Mean Absolute Error (MAE): 1.518720e-04

=== Weiner Deconvolution Results ===
Evaluation Time: 11.25 seconds
Mean Squared Error (MSE): 1.399276e-06
Mean Absolute Error (MAE): 4.848649e-04

=== RL Deconvolution Results ===
Evaluation Time: 84.48 seconds
Mean Squared Error (MSE): 4.794561e-06
Mean Absolute Error (MAE): 1.249856e-04
```

See the config at [this directory](./configs/deconv.yaml) to get more details of this task, but I used psf shear and 

TODO: There are some clear errors with the classical methods.