"""D4 data augmentation -- ABLATION ONLY.

This module exists purely to support the *augmentation-vs-architecture* ablation
for the paper. It is **not** an improvement to the flagship ``d4-fork-like``
model: because that model is exactly D4-equivariant by construction, augmenting
its training set with D4 transforms is provably a no-op (each transformed copy
produces a bit-identical loss and gradient), so it only wastes compute.

The one legitimate use is to *strengthen a non-equivariant baseline* (``cnn`` /
``fork-like``): applying the eight exact D4 transforms (90-degree rotations +
mirrors) with correctly spin-2-transformed labels gives 8x effective data from a
symmetry known to be exact. Showing that ``d4-fork-like`` still beats a
D4-*augmented* ``fork-like`` is the honest "baking the symmetry in beats
augmenting for it" comparison -- which is the only reason this code is here.

Gate it with ``dataset.d4_augment`` (default false). Do not enable it for
``d4-fork-like``.

The group action and the spin-2 label signs mirror ``core.models._d4_apply`` /
``_D4_W1`` / ``_D4_W2`` exactly, so the augmentation is consistent with the
equivariant model's own convention.
"""

import numpy as np

# Spin-2 sign of each shape component under the eight D4 elements i = 0..7,
# matching core.models._D4_W1 / _D4_W2 (kept as plain floats here so the
# augmentation runs on raw NumPy arrays during data prep, before JAX).
#   w1(g) = (-1)**r          (e1 is even under a mirror)
#   w2(g) = (-1)**(r + m)    (e2 flips under a mirror)
_W1 = [(-1.0) ** (i % 4) for i in range(8)]
_W2 = [(-1.0) ** ((i % 4) + (i // 4)) for i in range(8)]

# How each label key transforms under a D4 element: "e1"/"e2" follow the spin-2
# signs; every other quantity (size, flux, PSF size) is a D4 invariant scalar.
_SPIN2_E1 = {"g1", "psf_e1"}
_SPIN2_E2 = {"g2", "psf_e2"}


def _d4_apply_np(x, i):
    """Apply the ``i``-th D4 element to a batched image array ``(N, H, W)``.

    Mirrors :func:`shearnet.core.models._d4_apply` (rotations/flips on the two
    spatial axes) but in NumPy, since augmentation happens on raw stamps during
    data preparation.
    """
    r, m = i % 4, i // 4
    if m:
        x = np.flip(x, axis=1)
    if r:
        x = np.rot90(x, r, axes=(1, 2))
    return x


def _label_signs(output_keys):
    """Return an ``(8, n_keys)`` array of the spin-2 sign for each key/element."""
    signs = np.ones((8, len(output_keys)), dtype=np.float32)
    for j, key in enumerate(output_keys):
        if key in _SPIN2_E1:
            signs[:, j] = _W1
        elif key in _SPIN2_E2:
            signs[:, j] = _W2
        # else: invariant scalar -> stays +1
    return signs


def d4_augment(galaxy_images, psf_images, labels, output_keys):
    """Return the 8x D4-augmented copy of a (raw) training set. ABLATION ONLY.

    Applies each of the eight D4 group elements jointly to the galaxy and PSF
    stamps and applies the matching spin-2 sign flips to the labels. Operates on
    **raw** (un-normalized) arrays so it composes correctly with the label /
    image normalizers, which should be fit afterwards on the augmented set.

    Args:
        galaxy_images: ``(N, H, W)`` galaxy stamps.
        psf_images: ``(N, H, W)`` PSF stamps, or ``None`` for single-branch models.
        labels: ``(N, n_keys)`` raw labels, columns ordered as ``output_keys``.
        output_keys: tuple of label names (determines each column's spin).

    Returns:
        ``(gal_aug, psf_aug, labels_aug)`` with ``8 * N`` rows. ``psf_aug`` is
        ``None`` when ``psf_images`` is ``None``.
    """
    signs = _label_signs(output_keys)

    gal_aug, psf_aug, lab_aug = [], [], []
    for i in range(8):
        gal_aug.append(_d4_apply_np(galaxy_images, i))
        if psf_images is not None:
            psf_aug.append(_d4_apply_np(psf_images, i))
        lab_aug.append(labels * signs[i][None, :])

    gal_out = np.concatenate(gal_aug, axis=0)
    lab_out = np.concatenate(lab_aug, axis=0)
    psf_out = np.concatenate(psf_aug, axis=0) if psf_images is not None else None
    return gal_out, psf_out, lab_out
