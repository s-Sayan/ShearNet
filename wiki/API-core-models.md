# `shearnet.core.models`

> Module: `shearnet.core.models`

Flax neural-network architectures for galaxy shear estimation.

<a id="shearnet.core.models.ResidualBlock"></a>

## `ResidualBlock`

```python
class ResidualBlock(nn.Module)
```

A two-convolution residual block with a skip connection.

The input is passed through two convolutions and added back to a
(optionally 1x1-projected) copy of itself, following the residual-learning
design of He et al. (CVPR 2016). Used as a building block by ``GalaxyResNet``.

**Attributes**:

- `filters` - Number of output channels for the block.
- `kernel_size` - Convolution kernel size (default ``(3, 3)``).

<a id="shearnet.core.models.ResidualBlock.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x)
```

Apply the residual block to ``x`` and return the activated output.

<a id="shearnet.core.models.SimpleGalaxyNN"></a>

## `SimpleGalaxyNN`

```python
class SimpleGalaxyNN(nn.Module)
```

A plain multi-layer perceptron (``nn='mlp'``) for shear estimation.

Flattens the input stamp and applies dense layers to regress the requested
output parameters. The lightest of the available architectures.

The ``__call__`` signature is shared across all single-branch models:

**Arguments**:

- `x` - Input image batch, shape ``(batch, height, width)`` (a leading batch
  axis is added if a single 2-D stamp is passed).
- `deterministic` - Disables stochastic layers (e.g. dropout) when ``True``.
- `fork` - If ``True``, return the flattened feature vector instead of the
  final prediction (used by ``ForkLike`` to fuse two branches).
- `gap` - Use global-average-pooling instead of flattening (where supported).
- `output_keys` - Names of the parameters to predict; the output dimension
  equals ``len(output_keys)``.
  

**Returns**:

  Array of shape ``(batch, len(output_keys))`` with the predicted
  parameters, or the feature vector when ``fork=True``.

<a id="shearnet.core.models.SimpleGalaxyNN.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x,
             deterministic: bool = False,
             fork: bool = False,
             gap: bool = False,
             output_keys: tuple = ("g1", "g2"))
```

Run the MLP and return the predictions (or features when ``fork``).

<a id="shearnet.core.models.EnhancedGalaxyNN"></a>

## `EnhancedGalaxyNN`

```python
class EnhancedGalaxyNN(nn.Module)
```

A compact convolutional network (``nn='cnn'``) for shear estimation.

Two convolution + average-pool blocks extract spatial features which are
flattened and passed through dense layers. A good default architecture.

Shares the common model signature (see `SimpleGalaxyNN`). In addition,
``return_spatial=True`` returns the intermediate spatial feature map (used by
the transformer-fusion path of `ForkLike`).

<a id="shearnet.core.models.EnhancedGalaxyNN.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x,
             deterministic: bool = False,
             fork: bool = False,
             gap: bool = False,
             output_keys: tuple = ("g1", "g2"),
             return_spatial: bool = False)
```

Run the CNN and return predictions, features, or the spatial map.

<a id="shearnet.core.models.GalaxyResNet"></a>

## `GalaxyResNet`

```python
class GalaxyResNet(nn.Module)
```

A residual CNN (``nn='resnet'``) built from `ResidualBlock`s.

Applies an initial convolution followed by two residual blocks of growing
width, then dense layers with a ``tanh`` output. Heavier than
`EnhancedGalaxyNN`; shares the common model signature
(see `SimpleGalaxyNN`).

<a id="shearnet.core.models.GalaxyResNet.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x,
             deterministic: bool = False,
             fork: bool = False,
             gap: bool = False,
             output_keys: tuple = ("g1", "g2"))
```

Run the residual CNN and return predictions (or features).

<a id="shearnet.core.models.CBAM_Attention"></a>

## `CBAM_Attention`

```python
class CBAM_Attention(nn.Module)
```

Convolutional Block Attention Module with full citations.

<a id="shearnet.core.models.CBAM_Attention.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x)
```

Apply channel and spatial attention to ``x`` and return the result.

<a id="shearnet.core.models.EnhancedMultiScaleBlock"></a>

## `EnhancedMultiScaleBlock`

```python
class EnhancedMultiScaleBlock(nn.Module)
```

Enhanced multi-scale residual block with comprehensive citations.

<a id="shearnet.core.models.EnhancedMultiScaleBlock.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x, deterministic: bool = False)
```

Apply the multi-scale residual block to ``x`` and return the result.

<a id="shearnet.core.models.ResearchBackedGalaxyResNet"></a>

## `ResearchBackedGalaxyResNet`

```python
class ResearchBackedGalaxyResNet(nn.Module)
```

Research-backed Galaxy ResNet with comprehensive citations for every design decision.

OVERALL ARCHITECTURE PHILOSOPHY:
- Multi-scale processing: Inspired by galaxy morphology having features at different scales
- Residual learning: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
- Attention mechanisms: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)

<a id="shearnet.core.models.ResearchBackedGalaxyResNet.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x,
             deterministic: bool = False,
             fork: bool = False,
             gap: bool = False,
             output_keys: tuple = ("g1", "g2"),
             return_spatial: bool = False)
```

Run the multi-scale ResNet and return predictions, features, or map.

<a id="shearnet.core.models.ForkLensPSFNet"></a>

## `ForkLensPSFNet`

```python
class ForkLensPSFNet(nn.Module)
```

Strided CNN for PSF stamps (``nn='forklens_psf'``), from ForkLens.

Four stride-2 convolution blocks progressively downsample the PSF image.
Designed to be used as the PSF branch of `ForkLike`, mirroring the
``cnn_layers`` design of the ForkLens project. Shares the common model
signature (see `SimpleGalaxyNN`).

<a id="shearnet.core.models.ForkLensPSFNet.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(x,
             deterministic: bool = False,
             fork: bool = False,
             gap: bool = False,
             output_keys: tuple = ("g1", "g2"),
             return_spatial: bool = False)
```

Run the strided PSF CNN and return predictions, features, or map.

<a id="shearnet.core.models.TransformerFusion"></a>

## `TransformerFusion`

```python
class TransformerFusion(nn.Module)
```

Hybrid spatial cross-attention + self-attention fusion for ForkLike.

Galaxy spatial tokens act as queries; PSF spatial tokens act as keys/values.
This is physically motivated: the galaxy branch queries the PSF branch to
learn spatially-specific PSF correction.

**Arguments**:

- `d_model` - shared token dimension after projection (default 128)
- `num_heads` - attention heads (d_model must be divisible by this)
- `num_self_attn_layers` - self-attention layers applied after cross-attention

<a id="shearnet.core.models.TransformerFusion.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(galaxy_map,
             psf_map,
             output_keys: tuple = ("g1", "g2"),
             deterministic: bool = True)
```

Fuse galaxy and PSF feature maps and return the prediction.

<a id="shearnet.core.models.ForkLike"></a>

## `ForkLike`

```python
class ForkLike(nn.Module)
```

Combine two sub-models (galaxy and PSF branches) into one estimator.

Trains one sub-model on galaxy images and another on PSF images, then
concatenates their features and applies the dense/fully connected layers.

This is to mimimic the forklens structure here https://github.com/zhangzzk/forklens/.

<a id="shearnet.core.models.ForkLike.galaxy_model_type"></a>

#### `galaxy_model_type`

Default to EnhancedGalaxyNN

<a id="shearnet.core.models.ForkLike.psf_model_type"></a>

#### `psf_model_type`

Default to EnhancedGalaxyNN

<a id="shearnet.core.models.ForkLike.fusion"></a>

#### `fusion`

Options: "concat", "transformer"

<a id="shearnet.core.models.ForkLike.setup"></a>

#### `setup`

```python
def setup()
```

Initialize the sub-models during setup.

<a id="shearnet.core.models.ForkLike.__call__"></a>

#### `__call__`

```python
@nn.compact
def __call__(galaxy_image,
             psf_image,
             output_keys: tuple = ("g1", "g2"),
             deterministic: bool = False,
             gap: bool = False)
```

Run both branches, fuse their features, and return the prediction.
