"""Command-line interface for evaluating trained ShearNet models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import time
import optax
from flax.training import checkpoints, train_state

from ..config.config_handler import Config
from ..core.dataset import generate_dataset, split_combined_images
from ..core.models import (
    SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet,
    ResearchBackedGalaxyResNet, ForkLensPSFNet, ForkLike
)
from ..utils.normalization import load_normalizer, inverse_transform_labels

BOLD  = '\033[1m'
CYAN  = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
END   = '\033[0m'


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ShearNet model."
    )
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to load.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, defaults to saved training config).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config).')
    parser.add_argument('--test_samples', type=int, default=None,
                        help='Number of test samples (overrides config).')
    parser.add_argument('--mcal', action='store_true',
                        help='Also evaluate with NGmix metacalibration.')
    parser.add_argument('--plot', action='store_true',
                        help='Save learning curve and residual plots.')
    return parser


def load_config(args):
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')
    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)

    # Try to load the saved training config for this model
    model_config_path = os.path.join(default_plot_path, args.model_name, 'training_config.yaml')
    if args.config:
        config = Config(args.config)
    elif os.path.exists(model_config_path):
        print(f"\n{BOLD}Loading training config from: {model_config_path}{END}")
        config = Config(model_config_path)
    else:
        print(f"\nNo saved config found at {model_config_path}, using defaults.")
        config = Config()

    seed         = args.seed         if args.seed         is not None else config.get('evaluation.seed',         config.get('dataset.seed', 58))
    test_samples = args.test_samples if args.test_samples is not None else config.get('evaluation.test_samples', 1000)

    return {
        'model_name':    args.model_name,
        'seed':          seed,
        'test_samples':  test_samples,
        'psf_sigma':     config.get('dataset.psf_sigma') or config.get('dataset.psf_fwhm', 0.25),
        'nse_sd':        config.get('dataset.nse_sd',        1e-5),
        'exp':           config.get('dataset.exp',           'ideal'),
        'stamp_size':    config.get('dataset.stamp_size',    53),
        'pixel_size':    config.get('dataset.pixel_size',    0.141),
        'apply_psf_shear':  config.get('dataset.apply_psf_shear',  False),
        'psf_shear_range':  config.get('dataset.psf_shear_range',  0.05),
        'process_psf':   config.get('model.process_psf',    False),
        'nn':            config.get('model.type',            'cnn'),
        'galaxy_type':   config.get('model.galaxy.type',    'research_backed'),
        'psf_type':      config.get('model.psf.type',       'forklens_psf'),
        'n_outputs':     config.get('model.n_outputs',       2),
        'mcal':          args.mcal,
        'plot':          args.plot  or config.get('plotting.plot',   False),
        'psf_model':     config.get('comparison.psf_model', 'gauss'),
        'gal_model':     config.get('comparison.gal_model', config.get('dataset.type', 'gauss')),
        'save_path':     default_save_path,
        'plot_path':     default_plot_path,
    }


def generate_test_data(config):
    print(f"\n{BOLD}Generating {config['test_samples']} test galaxies...{END}")
    images, labels, obs = generate_dataset(
        config['test_samples'],
        config['psf_sigma'],
        exp=config['exp'],
        seed=config['seed'],
        nse_sd=config['nse_sd'],
        npix=config['stamp_size'],
        scale=config['pixel_size'],
        return_psf=config['process_psf'],
        return_obs=True,
        apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
    )

    if config['process_psf']:
        gal_images, psf_images = split_combined_images(images, has_psf=True, has_clean=False)
    else:
        gal_images = images
        psf_images = None

    print(f"  Galaxy images: {gal_images.shape}")
    print(f"  Labels:        {labels.shape}")
    return gal_images, psf_images, labels, obs


def load_model(config, gal_images, psf_images):
    print(f"\n{BOLD}Loading model '{config['model_name']}'...{END}")
    rng_key = random.PRNGKey(config['seed'])
    nn = config['nn']

    if nn == 'mlp':
        model = SimpleGalaxyNN()
    elif nn == 'cnn':
        model = EnhancedGalaxyNN()
    elif nn == 'resnet':
        model = GalaxyResNet()
    elif nn == 'research_backed':
        model = ResearchBackedGalaxyResNet()
    elif nn == 'forklens_psfnet':
        model = ForkLensPSFNet()
    elif nn == 'fork-like':
        model = ForkLike(
            galaxy_model_type=config['galaxy_type'],
            psf_model_type=config['psf_type']
        )
    else:
        raise ValueError(f"Unknown model type: {nn}")

    n_outputs = config['n_outputs']
    if config['process_psf']:
        init_params = model.init(rng_key, jnp.ones_like(gal_images[0]), jnp.ones_like(psf_images[0]))
    else:
        init_params = model.init(rng_key, jnp.ones_like(gal_images[0]))

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    # Find checkpoint
    load_path = config['save_path']
    matching = [
        d for d in os.listdir(load_path)
        if os.path.isdir(os.path.join(load_path, d)) and d.startswith(config['model_name'])
    ]
    if not matching:
        raise FileNotFoundError(
            f"No checkpoint found in {load_path} starting with '{config['model_name']}'"
        )
    model_dir = os.path.join(load_path, sorted(matching)[-1])
    print(f"  Loading checkpoint: {model_dir}")
    state = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=state)
    print(f"  {GREEN}✓ Loaded{END}")
    return state


def run_shearnet(state, gal_images, psf_images, labels, config, batch_size=128):
    """Run ShearNet predictions and compute basic metrics."""
    print(f"\n{BOLD}Running ShearNet predictions...{END}")
    start = time.time()

    # Load normalizer if present
    normalizer_path = os.path.join(
        config['plot_path'], config['model_name'], 'label_normalizer.npz'
    )
    norm_params = load_normalizer(normalizer_path) if os.path.exists(normalizer_path) else None

    preds_list = []
    n = len(gal_images)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        if config['process_psf']:
            batch_preds = state.apply_fn(
                state.params,
                gal_images[sl],
                psf_images[sl],
                deterministic=True,
            )
        else:
            batch_preds = state.apply_fn(
                state.params,
                gal_images[sl],
                deterministic=True,
            )
        preds_list.append(np.array(batch_preds))

    preds = np.concatenate(preds_list, axis=0)

    if norm_params is not None:
        preds = inverse_transform_labels(preds, norm_params)

    elapsed = time.time() - start

    # Metrics on g1, g2 only
    n_out = min(preds.shape[1], labels.shape[1], 2)
    mse  = float(np.mean((preds[:, :n_out] - labels[:, :n_out])**2))
    bias = float(np.mean(preds[:, :n_out] - labels[:, :n_out]))

    print(f"  MSE  (g1,g2): {YELLOW}{mse:.6e}{END}")
    print(f"  Bias (g1,g2): {YELLOW}{bias:+.6e}{END}")
    print(f"  Time:         {CYAN}{elapsed:.2f}s{END}")

    return preds, mse, bias, elapsed


def run_ngmix(obs, labels, config):
    """Run NGmix metacalibration and compute metrics."""
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred
    print(f"\n{BOLD}Running NGmix metacalibration...{END}")
    start = time.time()

    prior = _get_priors(config['seed'])
    rng   = np.random.RandomState(config['seed'])
    datalist, _ = mp_fit_one(
        obs, prior, rng,
        psf_model=config['psf_model'],
        gal_model=config['gal_model'],
    )
    preds = ngmix_pred(datalist)

    elapsed = time.time() - start

    # Filter NaNs
    valid = ~np.any(np.isnan(preds[:, :2]), axis=1)
    n_nan = np.sum(~valid)
    if n_nan:
        print(f"  Filtered {n_nan} NaN predictions.")

    mse  = float(np.mean((preds[valid, :2] - labels[valid, :2])**2))
    bias = float(np.mean(preds[valid, :2] - labels[valid, :2]))

    print(f"  MSE  (g1,g2): {YELLOW}{mse:.6e}{END}")
    print(f"  Bias (g1,g2): {YELLOW}{bias:+.6e}{END}")
    print(f"  Time:         {CYAN}{elapsed:.2f}s{END}")
    print(f"  Speedup vs ShearNet: measured separately above")

    return preds, mse, bias, elapsed


def print_summary(config, sn_mse, sn_bias, sn_time,
                  ngmix_mse=None, ngmix_bias=None, ngmix_time=None):
    print(f"\n{BOLD}{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}{END}")
    print(f"  Model:        {config['model_name']}")
    print(f"  Architecture: {config['nn']}")
    print(f"  Test samples: {config['test_samples']}")
    print(f"  Seed:         {config['seed']}")
    print(f"\n  {BOLD}ShearNet:{END}")
    print(f"    MSE:  {sn_mse:.6e}")
    print(f"    Bias: {sn_bias:+.6e}")
    print(f"    Time: {sn_time:.2f}s")
    if ngmix_mse is not None:
        print(f"\n  {BOLD}NGmix:{END}")
        print(f"    MSE:  {ngmix_mse:.6e}")
        print(f"    Bias: {ngmix_bias:+.6e}")
        print(f"    Time: {ngmix_time:.2f}s")
        print(f"    Speedup: {ngmix_time / sn_time:.1f}x")
    print(f"\n{GREEN}✓ Done{END}\n")


def main():
    parser = create_parser()
    args   = parser.parse_args()
    config = load_config(args)

    gal_images, psf_images, labels, obs = generate_test_data(config)
    state = load_model(config, gal_images, psf_images)
    sn_preds, sn_mse, sn_bias, sn_time = run_shearnet(
        state, gal_images, psf_images, labels, config
    )

    ngmix_mse = ngmix_bias = ngmix_time = None
    if config['mcal']:
        ng_preds, ngmix_mse, ngmix_bias, ngmix_time = run_ngmix(obs, labels, config)

    print_summary(config, sn_mse, sn_bias, sn_time,
                  ngmix_mse, ngmix_bias, ngmix_time)


if __name__ == '__main__':
    main()
