# ShearNet, everything on

The flagship ShearNet run: the fourth unit test's task, trained with every
feature the estimator has. One command:

    ./sub.sh fourth_shearnet

`d4-fork-like` is not a baseline. It is ShearNet's own architecture -- the
two-branch model whose branches are D4-equivariant by construction -- so this
directory is the ShearNet arm, not an ablation of one.

## What is on, and where it lives in the config

| feature | config | why it is on |
|---|---|---|
| D4 equivariance | `model.architecture: d4-fork-like`, `galaxy_branch: d4cnn` | Reynolds average over the eight square symmetries with the spin-2 sign weights, so a rotated stamp gives the correctly rotated shear |
| PSF equivariance | `model.process_psf: true`, `psf_branch: d4cnn` | the PSF goes through its own equivariant branch; rotating galaxy *and* PSF rotates the prediction |
| on-the-fly noise resampling | `train.generation: inloop` | the noise field is drawn inside the jitted step, so no galaxy is ever seen twice with the same realisation and there is no finite noise dataset to memorise |
| differentiable renderer | `train.backend: jax-galsim` | puts the renderer in the autodiff graph, which is what makes every term below a JVP rather than a finite difference |
| shear response target | `response.gamma_weight`, `gamma_target: analytic` | drives `R^gamma` to the exact per-object `1 - eps^2`, not the ensemble identity |
| PSF response | `response.psf_weight` | drives `R^PSF` to zero, which is exactly the right answer rather than an approximation |
| shift response | `response.shift_weight` | a shape estimate must not depend on where the object sits in the stamp |
| protected subspace | `response.complement_weight` | Hutchinson estimate of the Jacobian norm *outside* the physical tangents; the only bias-safe term, since it constrains directions no true signal lives in |
| PSF orbit | `response.orbit_weight`, `orbit_k: 4` | `{0,45,90,135}` degrees cancels spin-2 *and* spin-4 PSF leakage |
| EMA weights | `train.ema_decay: 0.999` | validate and checkpoint the averaged weights |
| response reporting | `response.report: true` | validation MSE cannot see a drifting `R^gamma` or growing PSF leakage; this is the actual selection signal |

`branch_features` is Lin et al. Table 1 (five layers at 32). The package
default is the two-layer proof of concept, so `research/unit_tests/fourth`
has been running a smaller backbone than the architecture it is named for.

## What is deliberately off

**Depth conditioning** (`train.noise.min_sd/max_sd/condition`). It is not the
same thing as the noise resampling above: resampling redraws the noise at a
*fixed* depth, conditioning randomises the depth itself and rescales both
inputs by the draw. The trainer rejects it outright alongside
`image.normalize_images: true`, and it would change what the run measures --
m would become an average over depths instead of the value at SuperBIT's. It
belongs in its own run.

**`base_shear_range`.** The COSMOS catalog shapes are real measured
ellipticities, already the physical shape distribution, so shearing them on
top moves the training population away from the realistic one. The response
terms do not need it: `base_g1`/`base_g2` are differentiable handles whatever
their value, so `R^gamma` is well defined at gamma = 0.

**`dropout`.** A genuine no-op for `d4cnn` branches -- only `research_backed`
consumes it -- so it is left at 0.0 rather than set for appearance.

**`d4_augment`.** Provably a no-op for an architecture that is already
equivariant, and the in-loop path rejects it. It exists for the
augmentation-vs-architecture ablation on the *non*-equivariant baselines.

## Cost

Measured per-step, all terms against the same step with no response terms
(`shearnet/core/inloop.py`, CPU, batch 8, fft 256 -- absolute times are not an
L40S but the ratios track the count of renders, JVPs and forwards, which is
device-independent):

    no response (control)                       1.00x
    only gamma / only psf / only shift          ~4.4x each
    only complement (12 image tangents)          4.44x
    only orbit, k=2                              2.21x
    only orbit, k=4                              4.23x
    all terms, every_n_steps=1, orbit_k=2       14.79x
    all terms, every_n_steps=1, orbit_k=4       16.73x
    all terms, every_n_steps=2, orbit_k=4        9.81x

Two things fall out of that table.

The single-term rows are all the same ~4.4x whichever term it is, because the
cost is the *linearisation of the renderer*, not the tangents applied to it.
gamma, psf and shift share one `jax.linearize`, so having all three costs
barely more than having one; `complement` pays a second linearisation of the
image map, which is why it is a separate 4.4x.

`orbit_k: 4` is the cheapest knob left: 14.79 -> 16.73 is **+13%** on top of
everything else, for the whole spin-4 leakage term. `every_n_steps` is the
expensive one: 1 -> 2 recovers **~40%**. If this run does not fit the queue,
drop the cadence to 2 first and leave `orbit_k` at 4 -- and watch the logged
`(N/M active steps)`, because the reported term values then average over a
subset of steps rather than all of them.

Caveat on the table: it was measured with a small stand-in backbone
(`(8,8,8)`, batch 8). The attempt to repeat it at this config's real width
(`(32,)*5`, batch 16) was killed during XLA compilation for lack of memory, so
the production multiplier is **not** measured -- and since the model forward
grows faster than the render does, the real multiplier should be *below* these
numbers, not above. Treat the table as an upper bound on the time cost and as
a warning about the memory cost: the terms hold a second linearisation of the
image map alive, so if the first thing this run does on the L40S is OOM, halve
`batch_size` and `jax_batch_size` together before touching anything else.

## Two caveats on `orbit_k: 4`, both measured

The 90-degree member is exact: it is a relabelling of a square grid, and the
PSF flux is conserved to float32. The 45 and 135-degree members are not, for
two separate reasons.

**The implicit pixel rotates too.** A PSFEx model is fitted to already-pixelised
stars, so its profile carries the pixel response -- that is why superbit draws
with `no_pixel`. Rotating it by 45 degrees rotates that pixel with it, so those
members sit slightly off the physical manifold.

**The interpolant resamples.** Rotating a PSFEx model 45 degrees samples it on
a grid that does not line up with its own square support. Measured over 40
distinct models: up to about **1% flux change per object, with zero mean across
the population** (-0.004%). Scatter, not a systematic -- which is the version
of this that does not bias the term.

Neither invalidates `orbit_k: 4`. What the term asserts is that a correct shear
estimate must not depend on which PSF the galaxy was convolved with, physically
realisable or not, and `eps^PSF` still rotates by exactly theta, which is all
the spin-4 cancellation argument needs. But if the logged orbit value turns out
to be dominated by the 45-degree members, these are the two things to suspect,
and `orbit_k: 2` is the clean fallback.

`tests/test_inloop.py::test_orbit_rotates_an_empirical_psf_too` pins all of
this on real PSFEx models at both K -- including that the 45-degree drift stays
zero-mean, since a systematic there would be a different situation. It skips
without the PSFEx files.

## Reading the result

Alone this run's m is uninterpretable -- it cannot be attributed to the
response terms rather than to the backbone or the in-loop noise. Two arms make
it interpretable, and both differ from this config in exactly one thing:

    fourth_forklens           equivariant: no    response: no
    fourth_shearnet_control   equivariant: yes   response: no
    fourth_shearnet           equivariant: yes   response: yes

Run them as three separate jobs, not one; each wants a whole GPU.
