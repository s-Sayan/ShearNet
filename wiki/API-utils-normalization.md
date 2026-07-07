# `shearnet.utils.normalization`

> Module: `shearnet.utils.normalization`

Label normalization utilities for ShearNet.

Z-score normalizes each output parameter independently so the loss
function weights all parameters equally regardless of magnitude
differences (e.g., g1/g2 ~ O(0.1) vs flux ~ O(1e4)).

Usage
-----
    from shearnet.utils.normalization import (
        fit_normalizer,
        transform_labels,
        inverse_transform_labels,
        save_normalizer,
        load_normalizer,
    )

    # Training
    norm_params = fit_normalizer(train_labels)
    train_labels_norm = transform_labels(train_labels, norm_params)
    save_normalizer(norm_params, path)

    # Evaluation
    norm_params = load_normalizer(path)
    preds_physical = inverse_transform_labels(preds_norm, norm_params)

<a id="shearnet.utils.normalization.fit_normalizer"></a>

#### `fit_normalizer`

```python
def fit_normalizer(labels: np.ndarray) -> dict
```

Compute per-parameter mean and std from labels.

Fits on the full array passed in, so pass only training labels
(not validation) to avoid distributional leakage.

Parameters
----------
labels : np.ndarray, shape (N, n_params)
    Raw (physical-unit) labels from generate_dataset.

Returns
-------
norm_params : dict
    {"mean": np.ndarray (n_params,), "std": np.ndarray (n_params,)}

<a id="shearnet.utils.normalization.transform_labels"></a>

#### `transform_labels`

```python
def transform_labels(labels: np.ndarray, norm_params: dict) -> np.ndarray
```

Z-score normalize labels to zero mean and unit variance.

Parameters
----------
labels : np.ndarray, shape (N, n_params)
norm_params : dict
    Output of fit_normalizer.

Returns
-------
np.ndarray, shape (N, n_params)
    Normalized labels.

<a id="shearnet.utils.normalization.inverse_transform_labels"></a>

#### `inverse_transform_labels`

```python
def inverse_transform_labels(labels_norm: np.ndarray,
                             norm_params: dict) -> np.ndarray
```

Denormalize predictions back to physical units.

Parameters
----------
labels_norm : np.ndarray, shape (N, n_params)
    Normalized predictions from the model.
norm_params : dict
    Output of fit_normalizer (or load_normalizer).

Returns
-------
np.ndarray, shape (N, n_params)
    Predictions in original physical units.

<a id="shearnet.utils.normalization.save_normalizer"></a>

#### `save_normalizer`

```python
def save_normalizer(norm_params: dict, path: str) -> None
```

Save normalization statistics to a .npz file.

Parameters
----------
norm_params : dict
    Output of fit_normalizer.
path : str
    Destination path, e.g. "<plot_path>/<model_name>/label_normalizer.npz".
    Parent directories are created automatically.

<a id="shearnet.utils.normalization.load_normalizer"></a>

#### `load_normalizer`

```python
def load_normalizer(path: str) -> dict
```

Load normalization statistics from a .npz file.

Parameters
----------
path : str
    Path to a file previously saved by save_normalizer.

Returns
-------
norm_params : dict
    {"mean": np.ndarray, "std": np.ndarray}
