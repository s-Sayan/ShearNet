# `shearnet.utils.plot_helpers`

> Module: `shearnet.utils.plot_helpers`

Plotting and visualization helpers for ShearNet.

Functions for learning curves, residual and true-vs-predicted plots, galaxy/PSF
stamp visualization, PSF-leakage diagnostics, and per-epoch animations.

<a id="shearnet.utils.plot_helpers.plot_learning_curve"></a>

#### `plot_learning_curve`

```python
def plot_learning_curve(losses, train_loss=None, path=None)
```

Plot loss over epochs.

<a id="shearnet.utils.plot_helpers.plot_residuals"></a>

#### `plot_residuals`

```python
def plot_residuals(true_labels,
                   predicted_labels,
                   path=None,
                   mcal=False,
                   preds_ngmix=None,
                   combined=False)
```

Plot the residuals (true - predicted) for both g1 and g2.

Optionally combine residuals for g1 and g2 into a single distribution.

<a id="shearnet.utils.plot_helpers.visualize_galaxy_samples"></a>

#### `visualize_galaxy_samples`

```python
def visualize_galaxy_samples(images,
                             true_labels,
                             predicted_labels,
                             snr_values,
                             num_samples=5,
                             path=None)
```

Visualize true and predicted labels on test images.

<a id="shearnet.utils.plot_helpers.visualize_psf_samples"></a>

#### `visualize_psf_samples`

```python
def visualize_psf_samples(images, num_samples=10, path=None)
```

Visualize PSF images with log-scaled colormap (shared scaling).

<a id="shearnet.utils.plot_helpers.plot_true_vs_predicted"></a>

#### `plot_true_vs_predicted`

```python
def plot_true_vs_predicted(true_labels,
                           predicted_labels,
                           path=None,
                           mcal=False,
                           preds_mcal=None)
```

Scatter true vs. predicted g1, g2, sigma and flux for the network.

**Arguments**:

- `true_labels` - Ground-truth array with columns ``(g1, g2, sigma, flux)``.
- `predicted_labels` - Network predictions in the same column order.
- `path` - If given, the figure is saved here; otherwise it is shown.
- `mcal` - If ``True``, overlay the moment/NGmix predictions in ``preds_mcal``.
- `preds_mcal` - Comparison predictions, required when ``mcal=True``.

<a id="shearnet.utils.plot_helpers.plot_residuals_v1"></a>

#### `plot_residuals_v1`

```python
def plot_residuals_v1(true_labels,
                      predicted_labels,
                      path=None,
                      mcal=False,
                      preds_ngmix=None,
                      combined=False)
```

Plot the residuals (true - predicted) for both g1 and g2.

Optionally combine residuals for g1 and g2 into a single distribution.

<a id="shearnet.utils.plot_helpers.plot_true_vs_predicted_v1"></a>

#### `plot_true_vs_predicted_v1`

```python
def plot_true_vs_predicted_v1(true_labels,
                              predicted_labels,
                              path=None,
                              mcal=False,
                              preds_mcal=None)
```

Plot true vs predicted values for both model and MCAL with residuals and error bars.

<a id="shearnet.utils.plot_helpers.plot_true_vs_predicted_anim"></a>

#### `plot_true_vs_predicted_anim`

```python
def plot_true_vs_predicted_anim(true_labels,
                                predicted_labels,
                                path=None,
                                mcal=False,
                                preds_mcal=None)
```

Plot true vs predicted values for both model and MCAL with residuals and error bars.

<a id="shearnet.utils.plot_helpers.animate_model_epochs"></a>

#### `animate_model_epochs`

```python
def animate_model_epochs(true_labels,
                         load_path,
                         plot_path,
                         epochs,
                         state,
                         model_name="model",
                         mcal=False,
                         preds_mcal=None)
```

Create an animation of the model predictions over different epochs.

<a id="shearnet.utils.plot_helpers.extract_psf_properties_from_obs"></a>

#### `extract_psf_properties_from_obs`

```python
def extract_psf_properties_from_obs(observations)
```

Extract PSF properties from ngmix observations for systematics analysis.

Parameters
----------
observations : list of ngmix.Observation
    List of observation objects

Returns
-------
psf_properties : dict
    Dictionary with 'e1', 'e2', 'T' arrays

<a id="shearnet.utils.plot_helpers.calculate_image_moments"></a>

#### `calculate_image_moments`

```python
def calculate_image_moments(image)
```

Calculate second moments and ellipticity from an image.

Parameters
----------
image : np.ndarray
    2D image array

Returns
-------
dict with keys:
    - 'e1': ellipticity component 1
    - 'e2': ellipticity component 2
    - 'T': size (trace of moment matrix)
    - 'flux': total flux

<a id="shearnet.utils.plot_helpers.plot_psf_systematics"></a>

#### `plot_psf_systematics`

```python
def plot_psf_systematics(predicted_shears,
                         psf_properties,
                         response_matrix=None,
                         datasets=None,
                         path=None,
                         title="PSF Systematics Analysis",
                         n_bins=20,
                         fit_method="linear")
```

Plot mean shear as a function of PSF properties (DES Y3 Figure 10 style).

Changes from original:
- Uses percentile-based binning for equal galaxy counts per bin
- Applies response correction to binned shear values
- Prints linear fit equations with uncertainties

Parameters
----------
predicted_shears : np.ndarray
    Predicted shear values, shape (N, 2) for [g1, g2]
psf_properties : dict
    Dictionary with 'e1', 'e2', 'T' for PSF properties
response_matrix : np.ndarray, optional
    2x2 response matrix [[R11, R12], [R21, R22]]. If provided,
    divides binned shears by response.
datasets : list of dict, optional
    Additional datasets for comparison
path : str, optional
    Path to save the plot
title : str
    Plot title
n_bins : int
    Number of bins (will have equal galaxy counts)
fit_method : str
    Fitting method (currently only 'linear' supported)

Returns
-------
fig : matplotlib.figure.Figure
results : dict
    Fit parameters and statistics

<a id="shearnet.utils.plot_helpers.plot_psf_systematics_from_eval"></a>

#### `plot_psf_systematics_from_eval`

```python
def plot_psf_systematics_from_eval(test_obs,
                                   predicted_labels,
                                   response_matrix=None,
                                   ngmix_preds=None,
                                   ngmix_response=None,
                                   path=None,
                                   n_bins=20)
```

Plot PSF systematics from the evaluation script (convenience wrapper).

This function is designed to integrate seamlessly with shearnet-eval CLI.

Parameters
----------
test_obs : list of ngmix.Observation
    Test observations containing PSF information
predicted_labels : np.ndarray
    Predicted shear values from neural network, shape (N, 4) [g1, g2, sigma, flux]
response_matrix : np.ndarray, optional
    2x2 response matrix for the neural network predictions
ngmix_preds : np.ndarray, optional
    Predictions from NGmix for comparison, shape (N, 4)
ngmix_response : np.ndarray, optional
    2x2 response matrix for NGmix predictions
path : str, optional
    Path to save the plot (without extension)
n_bins : int
    Number of bins (will have equal galaxy counts in each)

Returns
-------
fig : matplotlib.figure.Figure
    The figure object
results : dict
    Fit parameters and statistics

<a id="shearnet.utils.plot_helpers.calculate_psf_leakage_coefficients"></a>

#### `calculate_psf_leakage_coefficients`

```python
def calculate_psf_leakage_coefficients(predicted_shears,
                                       psf_properties,
                                       true_shears=None)
```

Calculate PSF leakage coefficients (α, β) from the systematic analysis.

Following DES Y3 methodology:
- α: PSF ellipticity leakage coefficient
- β: PSF size leakage coefficient

Parameters
----------
predicted_shears : np.ndarray
    Predicted shear values, shape (N, 2) for [g1, g2]
psf_properties : dict
    Dictionary with 'e1', 'e2', 'T' for PSF properties
true_shears : np.ndarray, optional
    True shear values to calculate residuals, shape (N, 2)

Returns
-------
coefficients : dict
    Dictionary containing leakage coefficients and their uncertainties
