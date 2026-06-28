# Contributing to ShearNet

Thanks for your interest in improving ShearNet! Contributions of all kinds (bug reports, documentation, and code) are welcome.

## Development setup

```bash
git clone https://github.com/s-Sayan/ShearNet.git
cd ShearNet
make install-dev          # conda env "shearnet_dev" with test/lint tools
conda activate shearnet_dev
# or, without conda:
pip install -e ".[dev]"
pip install git+https://github.com/esheldon/ngmix.git
```

## Running the tests

```bash
pytest tests/             # or: make test
```

The simulation/training tests need GalSim, NGmix, and JAX; tests that depend on
a missing optional dependency are skipped automatically.

A fast end-to-end smoke check (also run in CI):

```bash
shearnet-train --config configs/dry_run.yaml
shearnet-eval  --model_name dry_run
```

## Style

- Match the surrounding code style; format with `black` and `isort`
  (installed by `make install-dev`).
- Keep public functions documented with docstrings — the API wiki is generated
  from them via `python scripts/generate_wiki.py`.

## Submitting changes

1. Create a feature branch.
2. Make your change and add or update a test where it makes sense.
3. Ensure `pytest tests/` passes (or skips cleanly) and the dry-run config works.
4. Open a pull request describing the change and its motivation.
