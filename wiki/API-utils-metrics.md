# `shearnet.utils.metrics`

> Module: `shearnet.utils.metrics`

Metrics and evaluation functions for ShearNet.

This module contains functions for evaluating different shear estimation
methods including neural networks, NGmix, and metacalibration.

<a id="shearnet.utils.metrics.remove_nan_preds_multi"></a>

#### `remove_nan_preds_multi`

```python
def remove_nan_preds_multi(
        pred1: np.ndarray, pred2: np.ndarray,
        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Drop rows where either prediction array contains a NaN, keeping rows aligned.

<a id="shearnet.utils.metrics.remove_nan_preds"></a>

#### `remove_nan_preds`

```python
def remove_nan_preds(preds: np.ndarray,
                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

Drop rows with NaN predictions, returning the filtered ``(preds, labels)``.

<a id="shearnet.utils.metrics.loss_fn_mcal"></a>

#### `loss_fn_mcal`

```python
def loss_fn_mcal(images, labels, psf_fwhm)
```

MSE loss and per-component breakdown for moment-based metacalibration.

<a id="shearnet.utils.metrics.loss_fn_ngmix"></a>

#### `loss_fn_ngmix`

```python
def loss_fn_ngmix(obs_list,
                  labels,
                  seed=1234,
                  psf_model="gauss",
                  gal_model="gauss")
```

MSE loss, predictions and per-component bias for NGmix fits.

<a id="shearnet.utils.metrics.eval_ngmix"></a>

#### `eval_ngmix`

```python
def eval_ngmix(test_obs,
               test_labels,
               seed=1234,
               psf_model="gauss",
               gal_model="gauss") -> Dict[str, Any]
```

Run NGmix metacalibration and report loss, bias, timing and response matrix.

**Returns**:

- `dict` - Metrics including ``loss``, ``bias``, per-label breakdowns, the mean
  metacal response matrix ``R``, per-galaxy responses, predictions and timing.

<a id="shearnet.utils.metrics.eval_step"></a>

#### `eval_step`

```python
@jax.jit
def eval_step(state, images, labels)
```

Deterministic forward pass for a single-branch model; returns ``(loss, preds)``.

<a id="shearnet.utils.metrics.fork_eval_step"></a>

#### `fork_eval_step`

```python
@jax.jit
def fork_eval_step(state, images, psf_images, labels)
```

Deterministic forward pass for the ``fork-like`` model; returns ``(loss, preds)``.

<a id="shearnet.utils.metrics.eval_model"></a>

#### `eval_model`

```python
def eval_model(state,
               test_images,
               test_labels,
               output_keys=("g1", "g2"),
               batch_size=32) -> Dict[str, Any]
```

Evaluate a single-branch ShearNet model over a test set in batches.

**Returns**:

- `dict` - ``loss``, ``bias``, per-label MSE/bias, stacked predictions and timing.

<a id="shearnet.utils.metrics.fork_eval_model"></a>

#### `fork_eval_model`

```python
def fork_eval_model(state,
                    test_images,
                    test_psf_images,
                    test_labels,
                    output_keys=("g1", "g2"),
                    batch_size=32) -> Dict[str, Any]
```

Evaluate a two-branch ``fork-like`` model over a test set in batches.

**Returns**:

- `dict` - ``loss``, ``bias``, per-label MSE/bias, stacked predictions and timing.

<a id="shearnet.utils.metrics.calculate_response_matrix"></a>

#### `calculate_response_matrix`

```python
def calculate_response_matrix(state,
                              observations,
                              batch_size=32,
                              h=0.01,
                              model_type="standard",
                              psf_images=None)
```

Estimate the metacalibration response matrix R for a ShearNet model.

Uses the pre-rendered +/- e1/e2 sheared images stored in each observation's
``meta`` (see `shearnet.core.dataset.sim_func`) to finite-difference the
network's response to an applied shear.

**Arguments**:

- `state` - Trained ``TrainState`` to evaluate.
- `observations` - List of ngmix observations carrying the sheared-image meta.
- `batch_size` - Inference batch size.
- `h` - Applied shear step used to render the sheared images.
- `model_type` - ``'standard'`` (single branch) or ``'fork'`` (two branches).
- `psf_images` - PSF stamps, required when ``model_type='fork'``.
  

**Returns**:

- `tuple` - ``(R, R_per_galaxy)`` — the mean 2x2 response matrix and the
  per-galaxy responses.

<a id="shearnet.utils.metrics.calculate_multiplicative_bias"></a>

#### `calculate_multiplicative_bias`

```python
def calculate_multiplicative_bias(state,
                                  obs_g1_pos,
                                  obs_g1_neg,
                                  obs_g2_pos,
                                  obs_g2_neg,
                                  true_shear_step=0.02,
                                  batch_size=32,
                                  h=0.01,
                                  model_type="standard",
                                  psf_g1_pos=None,
                                  psf_g1_neg=None,
                                  psf_g2_pos=None,
                                  psf_g2_neg=None,
                                  R=None)
```

Estimate multiplicative (m) and additive (c) shear bias for a ShearNet model.

Predicts shears on galaxy sets sheared by +/- ``true_shear_step`` in each
component, calibrates them with the metacalibration response matrix, and fits
``gamma_est = (1 + m) * gamma_true + c`` per component.

**Returns**:

- `dict` - ``m1``, ``c1``, ``m2``, ``c2``, the per-dataset gamma estimates and
  the response matrix ``R`` used.

<a id="shearnet.utils.metrics.calculate_multiplicative_bias_ngmix"></a>

#### `calculate_multiplicative_bias_ngmix`

```python
def calculate_multiplicative_bias_ngmix(obs_g1_pos,
                                        obs_g1_neg,
                                        obs_g2_pos,
                                        obs_g2_neg,
                                        true_shear_step=0.02,
                                        h=0.01,
                                        seed=1234,
                                        psf_model="gauss",
                                        gal_model="gauss")
```

Compute NGmix shear bias (counterpart of `calculate_multiplicative_bias`).

Computes multiplicative (m) and additive (c) shear bias for the NGmix
metacalibration estimator on the same +/- sheared galaxy datasets, for an
apples-to-apples comparison against ShearNet.

<a id="shearnet.utils.metrics.get_admoms_ngmix_fit"></a>

#### `get_admoms_ngmix_fit`

```python
def get_admoms_ngmix_fit(obs: "ngmix.Observation",
                         reduced: bool = True) -> dict
```

Measure adaptive-moment ellipticity and size for an observation.

Fits adaptive moments with ngmix (for e1/e2) and GalSim HSM (for size), on a
flux-normalized copy of the image. Used to characterize PSF shape in
`shearnet.core.dataset.sim_func`.

**Arguments**:

- `obs` - The ngmix observation to fit.
- `reduced` - If ``True``, convert the distortion (e1, e2) to reduced shear
  (g1, g2) before returning.
  

**Returns**:

- `dict` - ``{"e1", "e2", "T", "flags"}`` where ``flags`` is non-zero if either
  fit failed or the image had no positive flux.
