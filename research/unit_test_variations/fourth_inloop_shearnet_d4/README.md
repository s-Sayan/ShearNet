# ShearNet-D4 branches, everything else held fixed

This is `fourth_inloop_shearnet` with the two branch backbones swapped from
`d4cnn` to `shearnet-d4`. Nothing else about the run changes -- same catalogue,
same SuperBIT PSFEx models, same in-loop JAX-GalSim renderer, same response,
orbit and isotropy terms at the same weights, same optimiser, same seed. The
only other edits are `meta.model_name` and `paths.root`, which have to name this
directory or the run would write over the control's checkpoint and plots.

```diff
 model:
   architecture: d4-fork-like
   process_psf: true
   fusion: transformer
   head: attention
-  galaxy_branch: d4cnn
-  psf_branch: d4cnn
+  galaxy_branch: shearnet-d4
+  psf_branch: shearnet-d4
   branch_features: [32, 32, 32, 32, 32]
```

So the difference between the two arms is the feature extractor and nothing
else, which is what makes the comparison readable.

## What changes inside the model

`d4cnn` is the backbone of Lin et al. (2026), five Conv/LayerNorm/GELU/avg-pool
layers. At a 53-pixel stamp those five pools take the map to **1x1** before
fusion, so the "spatial" cross-attention has a single token per branch and can
only condition one global galaxy vector on one global PSF vector.

`shearnet-d4` is the backbone of the ShearNet-D4 design report. Three residual
stages separated by two anti-aliased downsamplings:

| | galaxy | PSF |
|---|---|---|
| Stem | 53x53x32 | 53x53x32 |
| Stage 0 | 2 residual blocks | 1 residual block |
| Stage 1 | 26x26x48, 2 blocks | 26x26x48, 1 block |
| Stage 2 | 13x13x64, 1 block + dilated context block | 13x13x64, 1 block |

Fusion therefore happens at **13x13 = 169 tokens**, and the galaxy can attend to
distinct PSF regions. Selecting the branch also selects the rest of the report's
specification, because it specifies branches, fusion and heads as one
architecture: width-256 GELU feed-forward sublayers after each attention
sublayer, two-hidden-layer bias-free odd shape heads (256-128-128-1), and one
final linear layer per invariant scalar. The invariant-conditioned four-map
attention pooling the report asks for is what `head: attention` already does, so
that line is unchanged.

`branch_features` is left at `[32, 32, 32, 32, 32]`. It sizes the `d4cnn`
backbone only; `shearnet-d4` fixes its own widths (32/48/64), exactly as
`research_backed` and `forklens_psf` do.

Every quantity the response losses differentiate is still smooth: the backbone
is convolutions, LayerNorm, GELU and fixed linear pooling, with no ReLU kink, no
max pool and no batch statistic anywhere.

## Equivariance

Unchanged, and still exact. The Reynolds orbit average is spin-2 equivariant for
an *arbitrary* square-map backbone, so swapping the backbone cannot weaken it.
Measured at random initialisation on 53-pixel stamps in float32:

```
rot90  shape  1.0e-08     rot90  scalar  6.0e-07
mirror shape  1.1e-08     mirror scalar  6.0e-07
```

`tests/test_models.py` runs `("shearnet-d4", "shearnet-d4")` through the same
equivariance and head parametrisations as every other branch pair.

## Cost

The model goes from ~217k to ~667k parameters, but the parameter count is not
what to budget for. The activation cost is: the orbit puts `8 * batch_size`
images through two 53x53x32 residual stages instead of pooling to 26x26 in the
first layer, and the two attention matrices are `169^2` per head instead of `1`.
At `batch_size: 128` that is roughly 12-15 GB of forward activations before the
response terms add their re-renders. The `sub.sh` allocation
(`rtx_pro_6000_b`, 96 GB) has the room; if a different GPU does not, lower
`train.batch_size` rather than touching anything else, and say so when reporting
the comparison, since it is then no longer the only difference from the control.

## Running it

```bash
JAX_ENABLE_X64=1 shearnet-train --config \
  research/unit_test_variations/fourth_inloop_shearnet_d4/config.yaml
```

or on the cluster:

```bash
cd research/unit_test_variations
./sub.sh fourth_inloop_shearnet_d4
```
