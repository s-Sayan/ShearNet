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
pip install "git+https://github.com/AdamField118/JAX-GalSim@0213d84"
```

## Running the tests

```bash
pytest tests/             # or: make test
```

The simulation/training tests need GalSim, NGmix, and JAX; tests that depend on
a missing optional dependency are skipped automatically. A handful of
float64-agreement tests only run with `JAX_ENABLE_X64=1`.

A fast end-to-end smoke check (also run in CI):

```bash
shearnet-train --config configs/smoke.yaml --run runs/smoke
shearnet-eval  --run runs/smoke
```

## Things the tests hold you to

- **A new config key** goes in `shearnet/config/schema.py` (type, default, one
  line of meaning). Then `python scripts/make_docs.py` regenerates
  `docs/config.md`; `tests/test_docs.py` fails while it is stale.
- **A new catalog column** goes in `shearnet/io/catalog_schema.py`, with its
  unit and meaning; the same script regenerates `docs/catalog.md`. The catalog
  holds raw measurements only -- anything derived (responses, biases, cuts)
  belongs downstream.
- **The paper configs** are generated: edit `configs/paper/fiducial.yaml` or the
  arm's entry in `configs/paper/generate.py`, then run it. `--check` (and the
  test suite) catches a hand edit.

## Style

- Match the surrounding code style; format with `black` and `isort`
  (installed by `make install-dev`).
- Keep public functions documented with docstrings — they are the API reference,
  so keep them accurate.

## Submitting changes

1. Create a feature branch.
2. Make your change and add or update a test where it makes sense.
3. Ensure `pytest tests/` passes (or skips cleanly) and the smoke run works.
4. Open a pull request describing the change and its motivation.
