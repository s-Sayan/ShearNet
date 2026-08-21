# Full-featured in-loop ShearNet

This is the production ShearNet arm, separate from the existing control,
response-only, and ForkLens arms. It uses the smooth five-layer D4 two-branch
network with transformer fusion, so the galaxy/PSF pair is jointly
D4-equivariant and the response derivatives remain differentiable.

The fused JAX-GalSim step renders every batch in the training graph. It draws
fresh noise on every optimizer step, samples the noise standard deviation from
half to 1.5 times the nominal SuperBIT depth, and expresses both inputs in
noise units. It also enables the analytic shear target, zero PSF and shift
targets, the protected-subspace complement penalty, and the K=2 PSF-only orbit
constraint. The latter re-renders the galaxy against the PSF rotated by 90
degrees with shared noise; it is not an architectural averaging operation.

Run the training stage directly with:

```bash
JAX_ENABLE_X64=1 shearnet-train --config \
  research/unit_test_variations/fourth_inloop_shearnet/config.yaml
```

On the cluster, the existing submission helper also recognizes the variation:

```bash
cd research/unit_test_variations
./sub.sh fourth_inloop_shearnet
```

The latter submits training followed by the matched bias, leakage, and timing
jobs. It requires the batchable DES_PSFEx build of JAX-GalSim used by the other
in-loop SuperBIT variations.
