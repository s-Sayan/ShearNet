# Dev Notes

## Fork-like branch changes

I basically just made the [generate_dataset](./shearnet/core/dataset.py#L11) function return both the galaxy and psf images. Then I made all the functions in [metrics.py](./shearnet/utils/metrics.py) work with two datasets. I then adapted the [training script](./shearnet/core/train.py) to train both psf and galaxy models. I had to edit the [cli scipts](./shearnet/cli/) to work with the new [generate_dataset](./shearnet/core/dataset.py#L11) and [training scripts](./shearnet/core/train.py) along with the adapting both the [configs](./configs/) and [config scipts](./shearnet/config/) to properly give the information need to the [cli scipts](./shearnet/cli/).

### Results

I will write this after my meeting today.

## Next Steps

Same with this section.

## License

MIT License

## Contributing

Contributions welcome! Please submit issues or pull requests.