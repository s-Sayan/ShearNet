# shearnet-d4 with the D4-covariant relative 2D RoPE fusion encoding

`fourth_inloop_shearnet_d4` with one change: `model.fusion_pos` goes from
`learned` to `rope2d`. Separate `meta.model_name` and `paths.root` so the two
runs never share a checkpoint, a normalizer or an output FITS.

```diff
- fusion_pos: learned      # implicitly, before the key existed
+ fusion_pos: rope2d
-   every_n_steps: 2
+   every_n_steps: 4
```

## What changes in the model

The fusion cross-attention used a learned absolute embedding: one free vector
per raster cell, added to the tokens. It was the one component of this
architecture chosen by convention rather than derived from the problem, and it
is neither a function of the displacement nor D4 covariant.

`rope2d` derives the encoding instead, from three constraints:

1. **The observation is a convolution.** The PSF's contribution to the image at
   `p` depends only on the offset `d = p - p'`. Both maps reach the fusion
   through the same downsampling of the same centred stamp, so `d` is well
   defined between them. The encoding must therefore be relative, and must act
   on the projected queries and keys rather than on the tokens.
2. **D4 adds exactly one condition.** Under the 90-degree element
   `d = (du, dv) -> (-dv, du)`, so the rotated x-pair carries `-w*dv` and the
   rotated y-pair `w*du`: the original with the pairs swapped and one
   conjugated. That is realisable on the channel space by
   `rho(r): (X, Y) -> (conj(Y), X)`, a signed permutation of four real
   coordinates -- orthogonal, hence score-preserving -- **only if both pairs of
   a group share the frequency**. The usual axial RoPE, with independently
   tuned per-axis schedules, is not covariant.
3. **The grid sets the frequencies**, one to three cycles over the 12-cell
   extent. The base-10000 schedule of text models would put every high-k
   frequency below one cycle over the whole map, i.e. at the identity.

No free parameters are left over, and the encoding contributes none:
0.6143M -> 0.5921M.

`tests/test_models.py` pins all three: the score is invariant when both grids
translate together, covariance holds exactly with shared frequencies, and the
independent-per-axis control FAILS at the scale of the scores themselves --
which is what makes constraint 2 load-bearing rather than decorative.

## Why every_n_steps went 2 -> 4

The response block is ~90% of the training step and this is the only knob that
moves it much: 9.81x -> 6.35x of a plain step by the measurements in
`fourth_shearnet_k4/README.md`, about -35% per epoch, which is what brings 60
epochs inside a 24 h training job. `orbit_k` is not the lever (9.81 -> 9.0) and
stays at 4. `complement_weight` stays at 1e-3: it is the one response term with
no bias/variance trade against it.

## Relationship to `fourth_inloop_shearnet_d4`

That directory is now the **learned-encoding arm** (job 2237219). Note it is
*not* a clean single-variable ablation against this one -- `every_n_steps`
differs too -- so it serves as the pipeline validation at full scale, the
source of the `m_err` resolution estimate, and a fallback result. A matched
`learned` arm at `every_n_steps: 4` would be needed for the ablation table.
