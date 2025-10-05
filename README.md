# ShearNet

A JAX-based neural network implementation for galaxy shear estimation.

## Installation

### Quick Install

```bash
git clone https://github.com/s-Sayan/ShearNet.git
cd ShearNet

# CPU version
make install

# GPU version (CUDA 12)
make install-gpu

# Activate environment
conda activate shearnet  # or shearnet_gpu for GPU
```
### Manual Install

```bash
conda create -n shearnet python=3.11
conda activate shearnet
pip install -e .# or pip install -e ".[gpu]" for GPU
pip install git+https://github.com/esheldon/ngmix.git
python scripts/post_installation.py
```

## Usage

### Train a ShearNet model

```bash
shearnet-train --epochs 10 --batch_size 64 --samples 10000  --psf_sigma 0.25 --model_name cnn1 --plot --nn cnn --patience 20
```
or
```bash
shearnet-train --config ./configs/example.yaml
```
### Evaluate a ShearNet model

```bash
shearnet-eval --model_name cnn1 --test_samples 5000
```

### Train a DeconvNet model

```bash
deconvnet-train --config ./configs/decovnet_dry_run.yaml
```
### Evaluate a DeconvNet model

```bash
deconvnet-eval --model_name deconvnet_dry_run --plot
```

Key options:

- `-nn`: Model type (`mlp`, `cnn`, `resnet`, `research_backed`, `forklens_psf`, `fork-like`)
- `-mcal`: Compare with metacalibration and NGmix
- `-plot`: Generate plots

## Example Results

ShearNet provides shear estimates for g1, g2, sigma, and flux parameters. Example performance on test data:

### Comparison of predictions
On 5000 test samples of stamp size 53x53 and pixel size 0.141:
```
| Method                            | MSE (g1, g2)    | Time    |
|-----------------------------------|-----------------|---------|
| ShearNet (research backed)        | ~6.75e-6        | ~6.6s   |
| ShearNet (fork-like)              | ~4e-6           | ~2.5s   |
| Moment-based                      | ~1e-4           | ~142s   |
```

## Requirements

- Python 3.8+
- JAX (CPU/GPU)
- Flax, Optax
- GalSim, NGmix
- NumPy, SciPy, Matplotlib

See `pyproject.toml` for complete list.

## Repository Structure

```
ShearNet/
├── shearnet/
│   ├── core/       # Models, training, dataset
│   ├── methods/    # NGmix, moment-based
│   ├── utils/      # Metrics, plotting
│   ├── cli/        # Command-line tools
│   └── deconvnet/  # Deconvolution Nueral Networks
├── scripts/        # Setup scripts
├── Makefile        # Installation
└── pyproject.toml  # Dependencies

```

## Python API

```python
from shearnet.core.dataset import generate_dataset
from shearnet.core.train import train_modelv2
import jax.random as random

# Generate data
images, labels = generate_dataset(10000, psf_fwhm=0.8)

# Train
rng_key = random.PRNGKey(42)
state, train_losses, val_losses = train_model(
    images, labels, rng_key, epochs=50, nn='cnn'
)
```

## License

MIT License

## Contributing

Contributions welcome! Please submit issues or pull requests.