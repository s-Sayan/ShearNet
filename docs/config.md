# Configuration

One YAML file describes a run: who it is, how stamps are rendered, the model,
how it is trained, and what its evaluation measures. Both commands read the
same file:

```bash
shearnet-train --config my.yaml --run runs/my     # trains; saves the resolved config in the run
shearnet-eval  --run runs/my                      # evaluates with the run's own settings
```

Rules:

* A key that is not listed below is an **error** (with a suggestion), not a
  silently ignored line. So is a value of the wrong type, and a setting the
  chosen architecture would never read.
* Anything left out takes the default below. `shearnet-train` writes the full
  resolved config to `RUN/config.resolved.yaml`.
* Relative paths resolve against the directory of the YAML file that sets them.
* `1e-4` is a number; a key written twice is an error.
* `shearnet-eval --config OVERRIDE.yaml` may change only `evaluation.*`,
  `simulation.catalogs.eval_file` and `run_options.ncores`; the model, renderer
  and training population belong to the run.

`configs/example.yaml` is a short commented tour; `configs/paper/` holds the
paper campaign. Files in the two pre-schema layouts still load (translated, with
a warning per changed key); `python -m shearnet.config.legacy OLD.yaml` prints
the translation.

## Every setting

<!-- generated:start -->

### `run_options`

| key | type | default | meaning |
|---|---|---|---|
| `run_name` | str | null | Name of the run. Prefixes the evaluation catalog filename. |
| `outdir` | path or null | null | Run directory: model, normalizers, history, logs and every evaluation of this model live under it. `--run` on the command line overrides it. |
| `description` | str or null | null | Free text, carried into the run manifest. |
| `ncores` | int or null | null | Worker processes for GalSim rendering and the ngmix fits. null uses SLURM_CPUS_PER_TASK on a cluster and 1 elsewhere. |

### `simulation`

| key | type | default | meaning |
|---|---|---|---|
| `backend` | str: galsim / jax-galsim | `galsim` | Renderer. jax-galsim is differentiable and batched; in-loop training and the evaluation need it. |
| `pixel_scale` | float | `0.141` | Pixel scale. [arcsec / pixel] |
| `stamp_size` | int | `53` | Side of the square postage stamp. [pixel] |
| `noise_sigma` | float | `1e-05` | Standard deviation of the Gaussian pixel noise, in image counts. Not shape noise and not S/N. [count] |
| `gal_model` | str: exp / gauss | `exp` | Galaxy light profile. |
| `hlr_type` | str: constant / catalog | `constant` | constant: every galaxy has hlr 0.5 arcsec. catalog: the HLR column of the catalog. |
| `flux_type` | str: constant / catalog | `constant` | constant: every galaxy has flux 12258.97. catalog: the FLUX column of the catalog. |
| `apply_psf_shear` | bool | `False` | Draw a random per-object shear onto the PSF (an artificial transform of the PSF model, not a measured PSF ellipticity). |
| `psf_shear_range` | float | `0.05` | Half-width of the uniform PSF-shear draw when apply_psf_shear is on. |
| `jax_fft_size` | int | `256` | Pinned FFT grid of the jax-galsim renderer. [pixel] |
| `jax_batch_size` | int | `256` | Objects per batched jax-galsim render. Memory scales as jax_batch_size * jax_fft_size**2. |
| `psf.mode` | str: ideal / superbit | `ideal` | ideal: a round Gaussian of gaussian_fwhm. superbit: the PSFEx models under psfex_file. |
| `psf.gaussian_fwhm` | float | `0.5` | FWHM of the ideal Gaussian PSF. [arcsec] |
| `psf.psfex_file` | path or null | null | A PSFEx .psf file, or a directory of them, for mode superbit. null uses the library bundled with the repository. |
| `catalogs.train_file` | path or null | null | Catalog FITS with G1, G2, HLR and FLUX columns that the training galaxies are drawn from. null draws a synthetic population, which is only ever useful for tests. |
| `catalogs.eval_file` | path or null | null | Held-out catalog the evaluation galaxies are drawn from. Required whenever train_file is set: object i is row i whatever the seed, so a shared catalog would re-measure the training galaxies. |

### `model`

| key | type | default | meaning |
|---|---|---|---|
| `type` | str: mlp / cnn / resnet / research_backed / forklens_psfnet / fork-like / d4-fork-like | `cnn` | Architecture: mlp, cnn, resnet, research_backed, forklens_psfnet (single stamp), fork-like or d4-fork-like (galaxy + PSF). |
| `galaxy_branch` | str | `research_backed` | Galaxy branch of a two-branch model. |
| `psf_branch` | str | `forklens_psf` | PSF branch of a two-branch model. |
| `output_keys` | list[str] | `['g1', 'g2']` | Network outputs, in order. g1/g2 are the pre-PSF ellipticity of the rendered galaxy (intrinsic shape composed with the applied shear); hlr and flux are the profile inputs. |
| `gap` | bool | `False` | Global average pooling in the branches. |
| `fusion` | str: concat / transformer | `concat` | How the two branches are combined. |
| `fusion_pos` | str: learned / rope2d / none | `learned` | Positional encoding of the transformer fusion. |
| `head` | str: gap / attention | `gap` | Pooling head of d4-fork-like. |
| `dropout` | float | `0.0` | Spatial dropout of a research_backed branch. |
| `branch_features` | list[int] or null | null | Channel widths of a d4cnn branch. |
| `d4_features` | list[int] or null | null | Stage widths of a shearnet-d4 branch. |
| `d4_depths_galaxy` | list[int] or null | null | Residual blocks per stage, galaxy shearnet-d4 branch. |
| `d4_depths_psf` | list[int] or null | null | Residual blocks per stage, PSF shearnet-d4 branch. |
| `d4_multiscale` | bool or null | null | Dilated context block at the end of the galaxy shearnet-d4 branch. null keeps the architecture default (on). |
| `orbit_scan` | bool | `True` | Run the D4 orbit as a rematerialised scan (less memory, same parameters). |
| `design` | str or null: d4cnn / shearnet-d4 | null | d4-fork-like fusion/head layout: d4cnn or shearnet-d4. null infers it from the galaxy branch. |
| `d_model` | int or null | null | Fusion width. |
| `num_heads` | int or null | null | Fusion attention heads. |
| `num_pool_heads` | int or null | null | Attention pooling maps (head: attention). |
| `num_self_attn_layers` | int or null | null | Self-attention layers in the fusion block. |
| `ffn_dim` | int or null | null | Feed-forward width in the fusion block. |

### `training`

| key | type | default | meaning |
|---|---|---|---|
| `seed` | int | `42` | Seed of the training population and of the optimiser. |
| `nobj` | int | `10000` | Training objects (validation included). |
| `generation` | str: upfront / inloop | `upfront` | upfront: render the dataset before training. inloop: render inside the jitted step (needs jax-galsim). |
| `base_shear_range` | float | `0.0` | Half-width of a uniform per-object applied shear in the training population (jax-galsim only). |
| `epochs` | int | `10` | Maximum epochs. |
| `batch_size` | int | `32` | Batch size. |
| `learning_rate` | float | `0.001` | Peak learning rate. |
| `weight_decay` | float | `0.0001` | AdamW weight decay. |
| `patience` | int | `10` | Early stopping: validations without improvement. |
| `val_split` | float | `0.2` | Fraction of training.nobj held out for validation. |
| `eval_interval` | int | `1` | Validate every N epochs. |
| `loss` | str | `mse` | Training loss, a registry name (mse, mae, huber, ...). |
| `loss_weights` | list[float] or null | null | One weight per output key, in order. null weights them equally. |
| `ema_decay` | float or null | null | Exponential moving average of the weights; null is off. When on, validation and the saved model use the averaged weights. |
| `resample_noise` | bool | `False` | Upfront only: render noise-free and draw fresh noise every epoch. |
| `normalize_labels` | bool | `True` | Z-score the labels, fit on the training portion. |
| `normalize_images` | bool | `False` | Standardize the input stamps, fit on the training portion. |
| `d4_augment` | bool | `False` | 8x D4 augmentation of the training portion. An ablation control for non-equivariant models; upfront only. |
| `response.gamma_weight` | float | `0.0` | Drive R^gamma to its target. |
| `response.psf_weight` | float | `0.0` | Drive the PSF-shear response to zero. |
| `response.shift_weight` | float | `0.0` | Drive the translation response to zero. |
| `response.complement_weight` | float | `0.0` | Penalise sensitivity outside the physical tangents. |
| `response.orbit_weight` | float | `0.0` | PSF-orbit consistency under shared noise. |
| `response.isotropy_weight` | float | `0.0` | Penalise (R11 - R22)^2 + (R12 + R21)^2 of the residual. |
| `response.every_n_steps` | int | `1` | Evaluate the response terms every N steps. |
| `response.orbit_k` | int: 2 / 4 | `2` | PSF orbit size. |
| `response.gamma_target` | str: analytic / identity | `analytic` | analytic: the exact per-object derivative of the label. identity: I, an ensemble target. |
| `response.batch` | int | `0` | Evaluate the response terms on the first N objects of each batch (0 = all). |
| `response.report` | bool or null | null | Log the measured response matrices at every validation. null is on whenever a response weight is nonzero. |
| `noise.min_sd` | float or null | null | In-loop depth augmentation: lower noise bound. |
| `noise.max_sd` | float or null | null | In-loop depth augmentation: upper noise bound. |
| `noise.condition` | bool | `False` | Express the inputs in units of the sampled noise. |

### `evaluation`

| key | type | default | meaning |
|---|---|---|---|
| `seed` | int | `58` | Seed of the evaluation population. Must differ from training.seed. |
| `nobj` | int | `1000` | Objects per scene and ring station. |
| `estimators` | list[str] | `['shearnet', 'ngmix']` | What is measured on every stamp. |
| `batch_size` | int | `4096` | Stamps per ShearNet forward pass. |
| `scenes` | scenes | `zero, +/-0.01 on g1, +/-0.01 on g2` | Applied reduced shears to render, each {name, g1, g2}. Every scene is rendered from the same galaxies, offsets, PSFs and noise, so any two are pair-matched. |
| `rotations_deg` | list[float] | `[0.0]` | Ring stations: every scene is re-rendered with the intrinsic galaxy shape rotated by each of these angles. The PSF is not rotated. {0, 90} cancels the intrinsic shape; {0, 45, 90, 135} also the O(eps^2 g) term. [deg] |
| `metacal.psf` | str: dilate | `dilate` | Metacal reconvolution PSF. Only dilate has the *_psf products. |
| `metacal.step` | float | `0.01` | One-sided metacal shear step (ngmix 'step', SuperBIT 'mcal_shear'). The +/- products are 2*step apart. |
| `metacal.shearnet` | bool | `True` | Also run ShearNet on the nine metacal images ngmix fits. |
| `ngmix.gal_model` | str | `gauss` | ngmix galaxy model. |
| `ngmix.psf_model` | str | `gauss` | ngmix PSF model. |

<!-- generated:end -->
