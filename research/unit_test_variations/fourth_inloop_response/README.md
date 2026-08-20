The fourth unit test, moved onto the in-loop jax-galsim backend with every
differentiable response term switched on.

Everything that defines the fourth task is unchanged -- SuperBIT PSF, g1/g2/hlr/
flux from the catalog, 53px at 12.719674 noise, same seeds and catalogs. What
changes is how the network is trained:

- `generation: inloop` renders inside the jitted step, so noise is fresh every
  step and the renderer sits in the autodiff graph.
- All five response terms on: R^gamma to its analytic per-object target, R^PSF
  and the translation response to zero, the Hutchinson complement penalty, and
  the K=2 PSF-orbit consistency.
- `branch_features` set to the Lin et al. Table 1 width; fourth inherits the
  two-layer default.

This is the "everything on" arm. A good m here says the bundle works; it does
NOT say which term earned it. Read it against `fourth_inloop_control`.

REQUIRES: `JAX_ENABLE_X64=1`, and a jax-galsim with a batchable DES_PSFEx
(JAX-GalSim PR #261) pinned by COMMIT SHA -- in-loop + superbit needs it.
