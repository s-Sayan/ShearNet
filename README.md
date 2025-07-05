# Dev Notes

## My Model vs Main Branch Model 

I tweaked the model at [this link](https://github.com/s-Sayan/ShearNet/blob/main/shearnet/core/models.py#L43) based of numerous research papers. The model I refer to is [here](./shearnet/core/models.py#L323). Plotted here is the comparison of the original model vs my new model.

### Low Noise (nse_sd = 1e-5)

The comparison is also housed at [this directory](./notebooks/research_vs_control_low_noise/).

Here is the comparions plots:

![learning curve](./notebooks/research_vs_control_low_noise/learning_curves_comparison_20250702_172032.png)

![residuals comparison](./notebooks/research_vs_control_low_noise/residuals_comparison_20250702_172126.png)

![scatter comparison](./notebooks/research_vs_control_low_noise/prediction_comparison_20250702_172119.png)

### High Noise (nse_sd = 1e-3)

The comparison is also housed at [this directory](./notebooks/research_vs_control_high_noise/).

Here is the comparions plots:

![learning curve](./notebooks/research_vs_control_high_noise/learning_curves_comparison_20250702_191955.png)

![residuals comparison](./notebooks/research_vs_control_high_noise/prediction_comparison_20250702_192242.png)

![scatter comparison](./notebooks/research_vs_control_high_noise/residuals_comparison_20250702_192253.png)

## Next Steps

My next steps are to impliment psf images into the training data. This will chage the initial shape from (batch_size, 53, 53) to (batch_size, 53, 53, 2). I hope to also get noise images eventually as well. 

Training on this should only increase the accuracy of ShearNet, and adding both psf and noise images will put it on even ground with NGMix.

## License

MIT License

## Contributing

Contributions welcome! Please submit issues or pull requests.
