# ShearNet API Reference

This wiki is generated from the in-code docstrings with [pydoc-markdown](https://niklasrosenstein.github.io/pydoc-markdown/). Regenerate it with `python scripts/generate_wiki.py`.

## Core

- [`shearnet.core.dataset`](API-core-dataset) — Galaxy/PSF postage-stamp simulation and dataset generation for ShearNet.
- [`shearnet.core.models`](API-core-models) — Flax neural-network architectures for galaxy shear estimation.
- [`shearnet.core.train`](API-core-train) — Core training functions for ShearNet models.

## Command-line interface

- [`shearnet.cli.evaluate`](API-cli-evaluate) — Command-line interface for evaluating trained ShearNet models.
- [`shearnet.cli.train`](API-cli-train) — Command-line interface for training ShearNet models.

## Configuration

- [`shearnet.config.config_handler`](API-config-config_handler) — Layered YAML + command-line configuration handling for ShearNet.

## Methods (baselines)

- [`shearnet.methods.mcal`](API-methods-mcal) — Moment-based metacalibration shear estimation for ShearNet.
- [`shearnet.methods.ngmix`](API-methods-ngmix) — NGmix-based shear estimation and metacalibration utilities for ShearNet.

## Utilities

- [`shearnet.utils.device`](API-utils-device) — JAX device selection helpers for ShearNet.
- [`shearnet.utils.metrics`](API-utils-metrics) — Metrics and evaluation functions for ShearNet.
- [`shearnet.utils.normalization`](API-utils-normalization) — Label normalization utilities for ShearNet.
- [`shearnet.utils.plot_helpers`](API-utils-plot_helpers) — Plotting and visualization helpers for ShearNet.
- [`shearnet.utils.simutils`](API-utils-simutils) — GalSim WCS construction helpers for ShearNet simulations.
