# ShearNet with the two cost knobs turned up

`fourth_inloop_shearnet` is the flagship and is already trained. It has **every
feature on**: D4 equivariance, PSF equivariance, in-loop noise resampling, all
five response terms including the protected subspace, the analytic gamma
target, EMA, response reporting. Comparing the two configs on the 59 keys
training actually reads, exactly two differ, and both are *cost* knobs rather
than features:

| | `fourth_inloop_shearnet` | here |
|---|---|---|
| `response.every_n_steps` | 2 | **1** |
| `response.orbit_k` | 2 | **4** |

That is the entire difference. Nothing else about the model, the data, the
seeds or the eval changes, so the pair is a clean two-knob ablation.

## What the knobs buy

**`orbit_k: 4`** adds the 45 and 135-degree PSF-orbit members. K=2 already
cancels the whole linear (spin-2) PSF leakage, since `eps^PSF -> -eps^PSF` at
90 degrees; K=4 additionally cancels the spin-4 term `eps^2 conj(gamma)`,
because `sum exp(4 i theta)` vanishes over the four angles.

**`every_n_steps: 1`** applies the response terms on every gradient step
instead of every other one, so there is no aliasing between the supervised and
response gradients and the logged term values average over all steps rather
than half of them.

## What they cost

Per-step, measured against the same step with no response terms:

    no response (control)                       1.00x
    only gamma / only psf / only shift          ~4.4x each
    only complement (12 image tangents)          4.44x
    only orbit, k=2                              2.21x
    only orbit, k=4                              4.23x
    all terms, every_n_steps=1, orbit_k=2       14.79x
    all terms, every_n_steps=1, orbit_k=4       16.73x   <- here
    all terms, every_n_steps=2, orbit_k=4        9.81x
    all terms, every_n_steps=2, orbit_k=2       ~9.0x    <- fourth_inloop_shearnet

Every single-term row is the same ~4.4x whichever term it is, because the cost
is the *linearisation of the renderer* and not the tangents applied to it:
gamma, psf and shift share one `jax.linearize`, and `complement` pays a second
one for the image map. So `orbit_k: 4` is only **+13%** on top of everything
else, and `every_n_steps` is the expensive knob -- 1 -> 2 recovers **~40%**.

Roughly, this run is about **1.9x** the training cost of
`fourth_inloop_shearnet` for the spin-4 cancellation and the every-step
cadence. If that does not fit, take `orbit_k: 4` alone and leave the cadence at
2: it is nearly all of the physics for a tenth of the cost.

The table was measured with a small stand-in backbone (`(8,8,8)`, batch 8).
Repeating it at this config's real width was killed for memory during XLA
compilation, so treat it as an upper bound -- the model forward grows faster
than the render does, so the real multiplier should be lower.

## The caveat on K=4, measured

The 90-degree member is exact: a relabelling of a square grid, PSF flux
conserved to float32. The 45 and 135-degree members are not, for two reasons.

**The implicit pixel rotates too.** A PSFEx model is fitted to already-pixelised
stars, so its profile carries the pixel response -- which is why superbit draws
with `no_pixel`. Rotating it 45 degrees rotates that pixel with it.

**The interpolant resamples.** Rotating 45 degrees samples the model on a grid
that does not line up with its own square support. Over 40 distinct models: up
to ~1% flux change per object, **zero mean** across the population (-0.004%).
Scatter, not a systematic -- which is the version that does not bias the term.

Neither invalidates the term: what it asserts is that a correct shear estimate
must not depend on which PSF the galaxy was convolved with, physically
realisable or not, and `eps^PSF` still rotates by exactly theta, which is all
the spin-4 cancellation argument needs. But if the logged orbit value ends up
dominated by the 45-degree members, these are the two things to suspect.
`tests/test_inloop.py::test_orbit_rotates_an_empirical_psf_too` pins all of it
on real PSFEx models at both K.

## Running it

    ./sub.sh fourth_shearnet_k4

Three arms, each differing from the next in one thing:

    fourth_forklens           equivariant: no    response: no
    fourth_shearnet_control   equivariant: yes   response: no
    fourth_inloop_shearnet    equivariant: yes   response: yes  (k=2, n=2)
    fourth_shearnet_k4        equivariant: yes   response: yes  (k=4, n=1)

`fourth_shearnet_control` is a valid control for either ShearNet arm: with
every response weight at 0.0 the cadence and orbit size are inert.
