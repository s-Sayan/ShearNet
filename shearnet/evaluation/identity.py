"""Observation identities and independent random streams."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rng(identity, purpose):
    # RandomState accepts all eight words; do not truncate an identity to one seed.
    seed = np.frombuffer(bytes.fromhex(digest([identity, purpose])), dtype="<u4")
    return np.random.RandomState(seed)


def implementation():
    from importlib.metadata import PackageNotFoundError, version, distribution
    root = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))}
    packages = {}
    for name in ("numpy", "jax", "jaxlib", "JAX-GalSim", "galsim", "ngmix", "flax", "optax"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    external = {}
    for name, module in (("ngmix", "ngmix"), ("JAX-GalSim", "jax_galsim")):
        try:
            directory = Path(distribution(name).locate_file(module))
            external[name] = digest({str(p.relative_to(directory)): file_digest(p)
                                     for p in sorted(directory.rglob("*.py"))})
        except PackageNotFoundError:
            external[name] = None
    return {"source": digest(code), "packages": packages, "scientific_sources": external}


def observation_recipe(config, psf_files):
    simulation = dict(config.get("simulation"))
    simulation.pop("catalogs", None)
    simulation.pop("jax_batch_size", None)
    simulation["psf"] = dict(simulation["psf"])
    simulation["psf"].pop("psfex_file", None)
    catalog = config.get("simulation.catalogs.eval_file")
    return {
        "protocol": PROTOCOL, "simulation": simulation,
        "catalog": file_digest(catalog) if catalog else "synthetic",
        "psfs": [file_digest(p) for p in psf_files],
        "seed": config.get("evaluation.seed"),
        "scenes": config.get("evaluation.scenes"),
        "rotations": config.get("evaluation.rotations_deg"),
        "nobj": config.get("evaluation.nobj"),
        "metacal": {k: config.get(f"evaluation.metacal.{k}") for k in ("step", "psf")},
        "outputs": config.get("model.output_keys"),
    }


#: Number of fixed random +-1 patterns an image is projected on.
FINGERPRINT_PATTERNS = 3
_PATTERNS = {}


def _patterns(shape):
    if shape not in _PATTERNS:
        seed = np.frombuffer(hashlib.sha256(repr(shape).encode()).digest(), dtype="<u4")
        draw = np.random.RandomState(seed).randint(0, 2, (FINGERPRINT_PATTERNS,) + shape)
        _PATTERNS[shape] = 2.0 * draw - 1.0
    return _PATTERNS[shape]


def fingerprint(image):
    """``sum |x|`` and the projections of an image on fixed random +-1 patterns.

    The two evaluation jobs render on different devices (ShearNet on the GPU,
    ngmix on CPUs), so the same observation agrees between them only to
    roundoff, never bit for bit. Every entry is linear in the pixels (or bounded
    by ``sum |x|``), so roundoff moves it by ~1e-7 of ``sum |x|``. A different
    galaxy, noise field or PSF -- or the same image rotated, flipped or shifted
    -- moves the projections by about ``sum |x| / sqrt(npix)``, hundreds of times
    the merge's tolerance on a 53x53 stamp. Four float32 numbers per image.
    """
    x = np.asarray(image, dtype=np.float64)
    projections = np.tensordot(_patterns(x.shape), x, axes=x.ndim)
    return np.concatenate([[np.abs(x).sum()], projections]).astype(np.float32)
