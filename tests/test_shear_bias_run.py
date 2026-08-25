"""End-to-end smoke test of the training-matched benchmark entry point.

Every other test here checks a piece. This one trains a tiny model, writes the
artifacts where a real run writes them, and drives ``research/shear_bias/run.py``
the way the SLURM scripts do -- which is the only way to catch the wiring
between the saved config, the renderer, the estimators and the output file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galsim")
pytest.importorskip("anacal")
pytest.importorskip("ngmix")
jax = pytest.importorskip("jax")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "shear_bias"))

import run as harness  # noqa: E402

from shearnet.config.config_handler import Config  # noqa: E402

NPIX, SCALE, FWHM, MODEL = 53, 0.141, 0.5, "smoke_model"


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A real saved run: checkpoint, normalizer and training_config.yaml."""
    import yaml

    from shearnet.config.config_handler import load_default_config
    from shearnet.core.dataset import generate_dataset
    from shearnet.core.train import save_checkpoint, train_model
    from shearnet.utils.normalization import fit_normalizer, save_normalizer

    tmp_path = tmp_path_factory.mktemp("run")
    cfg = load_default_config()
    cfg["dataset"].update(
        {
            "samples": 32,
            "seed": 3,
            "stamp_size": NPIX,
            "pixel_size": SCALE,
            "psf_fwhm": FWHM,
            "nse_sd": 5.0,
            "exp": "ideal",
            # the evaluation needs jax-galsim for R^PSF; the model itself is
            # backend-agnostic, so this only changes which renderer draws the
            # benchmark stamps
            "backend": "jax-galsim",
        }
    )
    cfg["model"]["output_keys"] = ["g1", "g2", "hlr", "flux"]
    cfg["output"]["model_name"] = MODEL
    cfg["output"]["plot_path"] = str(tmp_path / "plots")
    cfg["output"]["save_path"] = str(tmp_path / "ckpt")
    model_dir = tmp_path / "plots" / MODEL
    model_dir.mkdir(parents=True)
    (tmp_path / "ckpt").mkdir()
    with open(model_dir / "training_config.yaml", "w") as handle:
        yaml.dump(cfg, handle)

    images, labels = generate_dataset(
        32, psf_fwhm=FWHM, npix=NPIX, scale=SCALE, seed=3, nse_sd=5.0,
        output_keys=("g1", "g2", "hlr", "flux"),
    )
    norm = fit_normalizer(labels)
    save_normalizer(norm, str(model_dir / "label_normalizer.npz"))
    state, *_ = train_model(
        images,
        (labels - norm["mean"]) / norm["std"],
        jax.random.PRNGKey(0),
        epochs=1,
        batch_size=8,
        nn="cnn",
        output_keys=("g1", "g2", "hlr", "flux"),
    )
    save_checkpoint(
        state, step=1, checkpoint_dir=str(tmp_path / "ckpt"), model_name=MODEL, overwrite=True
    )

    benchmark = {
        "meta": {"model_name": MODEL},
        "paths": {"root": str(tmp_path / "bench")},
        "eval": {
            "seed": 99,
            "n_obs": 16,
            "gal_model": "gauss",
            "evaluate": {
                "baseline": "both",
                "shear_true": 0.02,
                "n_jackknife": 4,
                "psf_model": "gauss",
                "response_step": 0.01,
                "component": "both",
                "shearnet_metacal": True,
                "shearnet_batch_size": 64,
                "output": "evaluation.fits",
            },
        },
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    with open(benchmark_path, "w") as handle:
        yaml.dump(benchmark, handle)
    return Config(str(benchmark_path)), Config(str(model_dir / "training_config.yaml"))


def test_evaluation_measures_every_estimator_under_every_correction(trained_run):
    """One pass, every estimator, every correction it admits, one file.

    The contract: an evaluation always carries ngmix, AnaCal's model fit and
    ShearNet, each under every correction that is defined for it. There is no
    configuration that produces a subset, so this asserts on the whole set
    rather than on whatever happened to be enabled.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)

    assert result["backend"] == "jax-galsim"
    assert result["seed"] == 99 and result["seed"] != training.get("dataset.seed")
    # the pairs the harness can actually measure: metacal shears images so it
    # needs ngmix's bootstrapper, and anacal is analytic so only AnaCal's own
    # fit has it. sim is universal.
    expected = {("ngmix", "metacal"), ("ngmix", "sim"), ("ngmix", "none"),
                ("anacal", "anacal"), ("anacal", "sim"), ("anacal", "none"),
                ("shearnet", "metacal"), ("shearnet", "sim"), ("shearnet", "none")}
    for estimator, correction in sorted(expected):
        prefix = f"{estimator}_{correction}"
        for field in ("m", "m_err", "c", "c_err"):
            assert np.isfinite(result[f"{prefix}_{field}"]), f"{prefix}_{field}"
        assert result[f"{prefix}_m_err"] > 0.0, prefix
        assert result[f"{prefix}_response"].shape == (2, 2)
        assert result[f"{prefix}_n_used"] > 0
    for estimator in harness.ESTIMATORS:
        assert result[f"{estimator}_leakage_psf_response"].shape == (2, 2)
    assert result["render_seconds"] > 0.0
    assert result["inference_seconds"] > 0.0


def test_the_baseline_switch_always_keeps_shearnet():
    """ShearNet is not optional; the switch chooses what it is measured against.

    Turning off the AnaCal fit is the point of the switch -- its fit is serial
    and costs ~7 h at production N -- so the run must still be a comparison
    after you do.
    """
    assert harness._estimators({}, None) == ["ngmix", "shearnet"]      # default
    assert harness._estimators({}, "ngmix") == ["ngmix", "shearnet"]
    assert harness._estimators({}, "anacal") == ["anacal", "shearnet"]
    assert harness._estimators({}, "both") == ["ngmix", "anacal", "shearnet"]
    assert harness._estimators({"baseline": "anacal"}, None) == ["anacal", "shearnet"]
    # the command line beats the config
    assert harness._estimators({"baseline": "anacal"}, "ngmix") == ["ngmix", "shearnet"]
    for bad in ("fpfs", "shearnet", ""):
        with pytest.raises(ValueError, match="baseline"):
            harness._estimators({}, bad)


def test_shearnet_keeps_metacal_when_ngmix_is_switched_off(trained_run):
    """metacal is a correction, not an estimator.

    ShearNet's metacal response comes from ngmix's bootstrapper, so dropping
    ngmix as a reported baseline must not silently cost ShearNet a column --
    that is exactly the half-result this harness refuses to produce.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("anacal", "shearnet"))
    for correction in ("metacal", "sim"):
        assert np.isfinite(result[f"shearnet_{correction}_m"]), correction
    assert np.isfinite(result["anacal_anacal_m"])
    # ngmix contributes no estimator columns. (`ngmix_nproc` stays: it is the
    # worker count metacal ran on, which is metadata about the run either way.)
    for correction in harness.CORRECTIONS:
        assert f"ngmix_{correction}_m" not in result, correction
    assert "ngmix_leakage_psf_response" not in result


def test_fpfs_is_out_of_the_evaluation_but_still_in_the_library():
    """FPFS is not an estimator any more; its measurement code still is."""
    assert "fpfs" not in harness.ESTIMATORS
    with pytest.raises(ValueError, match="unknown baseline"):
        harness._estimators({}, "fpfs")
    from shearnet.methods.anacal import fpfs_shapes, measure_fpfs, resmooth  # noqa: F401


def test_anacal_analytic_response_is_not_a_finite_difference(trained_run):
    """The analytic and simulation responses must be genuinely different code.

    Both estimate de/dgamma for the same fit on the same stamps -- anacal by
    differentiating through it with quintuple numbers, sim by re-rendering the
    scene. They should agree in the mean, which is the point of carrying both;
    they cannot be bit-identical, which would mean one is secretly the other.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("ngmix", "anacal", "shearnet"))
    analytic = result["anacal_anacal_response"]
    simulated = result["anacal_sim_response"]
    assert np.isfinite(analytic).all() and np.isfinite(simulated).all()
    assert not np.allclose(analytic, simulated), (analytic, simulated)
    # a well-behaved response on an isotropic population is near-diagonal
    assert abs(analytic[0, 0] - analytic[1, 1]) / abs(analytic[0, 0]) < 0.35, analytic


def test_evaluation_writes_one_fits_with_every_hdu(trained_run):
    from astropy.io import fits

    benchmark, training = trained_run
    harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    path = Path(benchmark.get("paths.root")) / benchmark.get("eval.evaluate.output")
    assert path.exists(), path

    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        assert names == ["PRIMARY", "TAB_P", "TAB_M", "TAB_P2", "TAB_M2",
                         "LEAKAGE", "SUMMARY", "BINNED", "LEAKSUM"], names
        n = benchmark.get("eval.n_obs")
        for hdu in ("TAB_P", "TAB_M", "TAB_P2", "TAB_M2", "LEAKAGE"):
            assert len(hdul[hdu].data) == n, hdu
        # the second pair is sheared in g2, which is what makes m2/c2 real
        assert hdul["TAB_P"].header["COMPONEN"] == 0
        assert hdul["TAB_P2"].header["COMPONEN"] == 1
        for column in ("g_th", "hlr_th", "flux_th", "gpsf", "Tpsf", "s2n",
                       "e_ngmix", "e_anacal", "e_shearnet",
                       "R_ngmix_metacal", "R_ngmix_sim",
                       "R_anacal_anacal", "R_anacal_sim",
                       "R_shearnet_metacal", "R_shearnet_sim",
                       # metacal measures its own noshear shape on the
                       # reconvolved stamp; that is what the metacal m/c
                       # divides, so it cannot be reconstructed from e_<est>
                       "e_ngmix_metacal", "e_shearnet_metacal",
                       # the network predicts size and flux too; shear_measure()
                       # slices those off, so they need their own forward pass
                       "hlr_shearnet", "flux_shearnet"):
            assert column in hdul["TAB_P"].data.names, (column, hdul["TAB_P"].data.names)
        for column in ("gpsf", "s2n", "e_ngmix", "e_ngmix_raw",
                       "e_shearnet", "e_shearnet_raw",
                       "Rpsf_ngmix_sim", "Rbar_psf_ngmix"):
            assert column in hdul["LEAKAGE"].data.names, column
        header_primary = hdul["PRIMARY"].header
        summary = hdul["SUMMARY"].data
        assert set(summary["component"]) == {0, 1}
        assert header_primary["COMPONEN"] == 0            # the primary component
        assert str(header_primary["MEASURED"]).strip() == "0,1"
        pairs = set(zip(summary["estimator"], summary["correction"]))
        assert {("ngmix", "metacal"), ("ngmix", "sim"),
                ("anacal", "anacal"), ("anacal", "sim"),
                ("shearnet", "metacal"), ("shearnet", "sim")} <= pairs, pairs
        # the binned m/c was being computed and dropped; it is per flux
        # quantile, each bin dividing by its own within-bin response
        binned = hdul["BINNED"].data
        assert len(binned) > 0
        assert pairs >= set(zip(binned["estimator"], binned["correction"]))
        assert set(binned["bin"]) == set(range(binned["bin"].max() + 1))
        leaksum = hdul["LEAKSUM"].data
        assert set(leaksum["estimator"]) == set(harness.ESTIMATORS)


def test_the_metacal_shapes_reproduce_the_summary_row(trained_run):
    """SUMMARY must be recomputable from the per-object columns.

    The metacal m/c divides metacal's own noshear shape, not the plain
    measurement in ``e_<est>``. Storing only the response made the file
    unable to explain its own summary row, which is what this pins.
    """
    from astropy.io import fits

    from shearnet.methods.anacal import ShapeMeasurement, paired_bias

    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    path = Path(benchmark.get("paths.root")) / benchmark.get("eval.evaluate.output")
    with fits.open(path) as hdul:
        tabs = {"plus": hdul["TAB_P"].data, "minus": hdul["TAB_M"].data}
        header = hdul["PRIMARY"].header

        def shape(which, estimator, correction, column):
            tab = tabs[which]
            flag = f"flag_{estimator}"
            return ShapeMeasurement(
                e=np.asarray(tab[column], float)[:, :2],
                dedg=np.asarray(tab[f"R_{estimator}_{correction}"], float),
                flags=np.asarray(tab[flag], float) != 0 if flag in tab.names
                else np.zeros(len(tab), bool),
            )

        for estimator, correction, column in (
            ("ngmix", "metacal", "e_ngmix_metacal"),
            ("shearnet", "metacal", "e_shearnet_metacal"),
            ("anacal", "anacal", "e_anacal"),
            ("ngmix", "sim", "e_ngmix"),
        ):
            bias = paired_bias(
                shape("plus", estimator, correction, column),
                shape("minus", estimator, correction, column),
                float(header["SHEAR_TR"]),
                component=int(header["COMPONEN"]),
                njac=int(header["N_JACKKN"]),
                c_convention=str(header["C_CONVEN"]).strip(),
                resample=str(header["RESAMPLE"]).strip(),
            )
            prefix = f"{estimator}_{correction}"
            assert bias.m == pytest.approx(result[f"{prefix}_m"], rel=1e-9), prefix
            assert bias.c == pytest.approx(result[f"{prefix}_c"], rel=1e-9), prefix


def test_psf_response_is_applied_to_deconvolvers_only(trained_run):
    """ngmix's leakage is R^PSF-corrected; ShearNet's is deliberately not.

    The metacal-family PSF response equals physical leakage only for an
    estimator that explicitly deconvolves. Applying ShearNet's would fold the
    network's own PSF sensitivity into the number the leakage plot exists to
    show, so the default corrects ngmix and anacal and leaves ShearNet raw.
    """
    from astropy.io import fits

    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section["psf_response_apply"] = ["ngmix", "anacal"]
    try:
        result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    finally:
        section.clear()
        section.update(saved)
    assert result["ngmix_leakage_corrected"] is True
    assert result["anacal_leakage_corrected"] is True
    assert result["shearnet_leakage_corrected"] is False

    path = Path(benchmark.get("paths.root")) / benchmark.get("eval.evaluate.output")
    with fits.open(path) as hdul:
        leak = hdul["LEAKAGE"].data
        gpsf = np.asarray(leak["gpsf"], float)
        rbar = np.asarray(leak["Rbar_psf_ngmix"], float)
        assert np.allclose(rbar, rbar[0])
        # the correction is the constant ensemble response, not the per-object one
        assert np.allclose(
            np.asarray(leak["e_ngmix"], float),
            np.asarray(leak["e_ngmix_raw"], float) - gpsf * rbar[0],
            equal_nan=True,
        )
        # ShearNet's corrected column IS its raw column
        assert np.allclose(
            np.asarray(leak["e_shearnet"], float),
            np.asarray(leak["e_shearnet_raw"], float),
            equal_nan=True,
        )


def test_shearnet_metacal_can_be_switched_off(trained_run):
    """The out-of-distribution diagnostic is optional; the physical one is not.

    Turning it off must cost ShearNet only its metacal column -- ``sim`` is its
    response, and ngmix keeps its own metacal either way.
    """
    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section["shearnet_metacal"] = False
    try:
        result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
    finally:
        section.clear()
        section.update(saved)
    assert "shearnet_metacal_m" not in result
    assert np.isfinite(result["shearnet_sim_m"])
    assert np.isfinite(result["ngmix_metacal_m"])
    # still measured and reported, just not applied
    assert result["shearnet_leakage_psf_response"].shape == (2, 2)


def test_psf_response_apply_switch():
    # nothing is corrected by default: on job 2180809 subtracting the
    # finite-difference R^PSF turned ngmix's raw leakage slope of -0.002 into
    # -0.457 at 50 sigma. R^PSF and the regression slope alpha are different
    # quantities and only alpha may be subtracted.
    assert harness._psf_response_apply({}) == frozenset()
    assert harness._psf_response_apply({"psf_response_apply": "none"}) == frozenset()
    assert harness._psf_response_apply({"psf_response_apply": ["ngmix"]}) == frozenset({"ngmix"})
    assert harness._psf_response_apply({"psf_response_apply": "ngmix, shearnet"}) == frozenset(
        {"ngmix", "shearnet"}
    )
    assert harness._psf_response_apply({"psf_response_apply": "all"}) == frozenset(harness.ESTIMATORS)
    with pytest.raises(ValueError, match="psf_response_apply"):
        harness._psf_response_apply({"psf_response_apply": ["fpfs"]})


def test_shearnet_metacal_measures_the_reconvolved_images(trained_run):
    """ShearNet's metacal response must come from metacal's own images.

    Not from the scene protocol wearing a different name: the two are different
    derivatives and the whole point of reporting both is that they can disagree.
    The check is that the network's metacal response is NOT identical to its
    anacal one, and that both are finite.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    metacal = result["shearnet_metacal_response"]
    anacal = result["shearnet_sim_response"]
    assert np.isfinite(metacal).all() and np.isfinite(anacal).all()
    assert not np.allclose(metacal, anacal), (metacal, anacal)


def test_the_backend_requirement_fails_before_the_expensive_part(trained_run, monkeypatch):
    """A galsim-backend run cannot measure R^PSF, so it must not start at all."""
    import yaml

    from shearnet.config.config_handler import Config

    benchmark, training = trained_run
    path = Path(training.config["output"]["plot_path"]) / MODEL / "galsim_config.yaml"
    with open(path, "w") as handle:
        yaml.dump({**training.config, "dataset": {**training.get("dataset"), "backend": "galsim"}}, handle)

    import shearnet.benchmarking as bm

    calls = {"n": 0}
    real = bm.TrainingMatchedRenderer.render
    monkeypatch.setattr(
        bm.TrainingMatchedRenderer, "render",
        lambda self, *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real(self, *a, **kw))[1],
    )
    with pytest.raises(ValueError, match="jax-galsim"):
        harness._run_evaluation(benchmark, Config(str(path)), harness.ESTIMATORS)
    assert calls["n"] == 0, "it rendered before deciding it could not finish"


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        harness._estimators({}, "metacal")
    assert harness._estimators({"baseline": "ngmix"}, None) == ["ngmix", "shearnet"]


def test_both_components_are_measured(trained_run):
    """m and c exist for g1 AND g2, and they are separate measurements.

    Shearing only g1 and then asking paired_bias for component 1 does not give
    m2 -- the numerator is consistent with zero, so it returns roughly -1. A
    real m2 needs a second pair sheared in g2, which is what `component: both`
    renders.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
    for estimator in ("ngmix", "shearnet"):
        for correction in ("metacal", "sim"):
            for k in (1, 2):
                prefix = f"{estimator}_{correction}_g{k}"
                for field in ("m", "m_err", "c", "c_err"):
                    assert np.isfinite(result[f"{prefix}_{field}"]), f"{prefix}_{field}"
            # the two components are genuinely different measurements
            assert (result[f"{estimator}_{correction}_g1_m"]
                    != result[f"{estimator}_{correction}_g2_m"])
        # the unsuffixed alias is the first measured component, so every
        # existing caller keeps working
        assert result[f"{estimator}_sim_m"] == result[f"{estimator}_sim_g1_m"]


def test_component_switch():
    assert harness._components({}) == (0,)
    assert harness._components({"component": 1}) == (1,)
    assert harness._components({"component": "both"}) == (0, 1)
    assert harness._components({"component": [1, 0]}) == (1, 0)
    assert harness._components({"component": [0, 0]}) == (0,)
    for bad in (2, -1, "g1"):
        with pytest.raises(ValueError, match="component"):
            harness._components({"component": bad})


def test_shearnet_metacal_is_off_by_default(trained_run):
    """The network's reported response is the direct one unless asked otherwise.

    metacal feeds it reconvolved stamps that are out of its training
    distribution, so it is opt-in. ngmix keeps its metacal regardless.
    """
    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section.pop("shearnet_metacal", None)
    section["component"] = 0
    try:
        result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
    finally:
        section.clear()
        section.update(saved)
    assert "shearnet_metacal_m" not in result
    assert np.isfinite(result["shearnet_sim_m"])
    assert np.isfinite(result["ngmix_metacal_m"])


# ----------------------------------------------------------------------
# the truth-referenced ("none") row
# ----------------------------------------------------------------------
def test_the_sim_row_is_a_nonlinearity_check_and_the_none_row_is_the_calibration():
    """A perfectly linear estimator with a 10% bias must show m_sim = 0.

    This is the whole reason the ``none`` correction exists, so it is asserted
    on a case where the answer is known exactly rather than inferred from a run.

    ``paired_bias`` forms ``<(e+ - e-)/2> / <R>`` and divides by the applied
    shear. Substituting the ``sim`` response makes the numerator the secant of
    the response curve over ``+/-gamma`` and the denominator its secant over
    ``+/-(gamma + step)``. For a linear estimator those secants are the SAME
    number whatever the slope is, so the ratio is 1 and m is 0 -- no matter how
    badly the estimator is calibrated. Dividing by the identity instead compares
    against the applied shear, which is an input to the render, and recovers the
    bias.
    """
    from shearnet.methods.anacal import ShapeMeasurement, paired_bias

    rng = np.random.default_rng(0)
    n, gamma, step, slope = 20_000, 0.01, 0.01, 0.9  # a real -10% bias
    intrinsic = rng.normal(scale=0.25, size=(n, 2))

    def measure(g1, g2):
        return intrinsic + slope * np.array([g1, g2])

    def sim_response(base):
        columns = []
        for axis in (0, 1):
            high = list(base); high[axis] += step
            low = list(base); low[axis] -= step
            columns.append((measure(*high) - measure(*low)) / (2.0 * step))
        return np.stack(columns, axis=-1)

    flags = np.zeros(n, dtype=bool)
    identity = np.broadcast_to(np.eye(2), (n, 2, 2))
    plus, minus = measure(gamma, 0.0), measure(-gamma, 0.0)

    sim = paired_bias(
        ShapeMeasurement(e=plus, dedg=sim_response([gamma, 0.0]), flags=flags),
        ShapeMeasurement(e=minus, dedg=sim_response([-gamma, 0.0]), flags=flags),
        gamma, component=0, njac=20,
    )
    none = paired_bias(
        ShapeMeasurement(e=plus, dedg=identity, flags=flags),
        ShapeMeasurement(e=minus, dedg=identity, flags=flags),
        gamma, component=0, njac=20,
    )
    # blind to a 10% bias, to machine precision
    assert abs(sim.m) < 1e-12, sim.m
    assert np.isclose(sim.response[0, 0], slope)
    # and the truth-referenced row recovers it exactly
    assert np.isclose(none.m, slope - 1.0), none.m
    assert np.isclose(none.response[0, 0], 1.0)


def test_the_none_row_divides_by_the_identity_and_reproduces_from_the_table(trained_run):
    """SUMMARY's ``none`` rows are recomputable from TAB_P/TAB_M by hand."""
    from astropy.io import fits

    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
    shear = benchmark.get("eval.evaluate.shear_true")

    for estimator in ("ngmix", "shearnet"):
        assert np.allclose(result[f"{estimator}_none_response"], np.eye(2))

    path = Path(benchmark.get("paths.root")) / benchmark.get("eval.evaluate.output")
    with fits.open(path) as hdul:
        plus, minus = hdul["TAB_P"].data, hdul["TAB_M"].data
        summary = hdul["SUMMARY"].data
        for estimator in ("ngmix", "shearnet"):
            e_p = np.asarray(plus[f"e_{estimator}"], dtype=float)
            e_m = np.asarray(minus[f"e_{estimator}"], dtype=float)
            keep = np.isfinite(e_p).all(axis=1) & np.isfinite(e_m).all(axis=1)
            by_hand = np.nanmean(0.5 * (e_p[keep, 0] - e_m[keep, 0])) / shear - 1.0
            row = summary[(summary["estimator"] == estimator)
                          & (summary["correction"] == "none")
                          & (summary["component"] == 0)]
            assert len(row) == 1, estimator
            assert np.isclose(row["m"][0], by_hand, rtol=1e-10), estimator


# ----------------------------------------------------------------------
# shape-noise cancellation
# ----------------------------------------------------------------------
def test_the_ring_parser_spells_out_the_cost():
    """`true` is the 90-degree pair; the full ring has to be asked for by count."""
    assert harness._rotations({}) == (0.0,)
    assert harness._rotations({"shape_noise_cancel": False}) == (0.0,)
    assert harness._rotations({"shape_noise_cancel": True}) == (0.0, 90.0)
    assert harness._rotations({"shape_noise_cancel": 2}) == (0.0, 90.0)
    assert harness._rotations({"shape_noise_cancel": 4}) == (0.0, 45.0, 90.0, 135.0)
    assert harness._rotations({"shape_noise_cancel": 1}) == (0.0,)
    for bad in (3, 8, "yes"):
        with pytest.raises(ValueError, match="shape_noise_cancel"):
            harness._rotations({"shape_noise_cancel": bad})


def test_45_degrees_cancels_the_term_that_90_degrees_cannot():
    """The ring choice is a measurement, not a preference, so it is asserted.

    The plus/minus difference of the Moebius composition is
    ``gamma - eps^2 conj(gamma)``, not ``gamma``. A 90-degree rotation sends
    ``eps -> -eps`` and leaves ``eps^2`` untouched, so a {0, 90} ring cannot
    remove that term however many galaxies it averages. A 45-degree rotation
    sends ``eps -> i eps``, so ``eps^2 -> -eps^2``, and the term cancels object
    by object.
    """
    rng = np.random.default_rng(1)
    n, gamma = 400_000, 0.01 + 0j
    eps = rng.normal(scale=0.25, size=n) + 1j * rng.normal(scale=0.25, size=n)

    def compose(e, g):                       # Seitz & Schneider (1997)
        return (e + g) / (1.0 + np.conj(g) * e)

    def m_none(phases):
        difference = np.mean(
            [0.5 * (compose(p * eps, gamma) - compose(p * eps, -gamma)) for p in phases],
            axis=0,
        )
        return (difference.mean() / gamma).real - 1.0, np.std(difference.real) / abs(gamma)

    m_pair, scatter_pair = m_none([1, -1])          # {0, 90}
    m_ring, scatter_ring = m_none([1, 1j, -1, -1j])  # {0, 45, 90, 135}

    # 90 degrees leaves the bias exactly where it was
    assert abs(m_pair) > 1e-4, m_pair
    assert scatter_pair > 0.1, scatter_pair
    # 45 degrees removes it, and removes the scatter with it -- per object, down
    # to the O(gamma^3 eps^4) term the four-station ring does not reach (5.4e-06
    # here, against 0.125 for the pair: a factor of 23000)
    assert abs(m_ring) < 1e-6, m_ring
    assert scatter_ring < 1e-5, scatter_ring
    assert scatter_ring < 1e-4 * scatter_pair, (scatter_ring, scatter_pair)


def test_average_rotations_averages_shapes_responses_and_unions_flags():
    from shearnet.methods.anacal import ShapeMeasurement

    def measurement(value, flag):
        return ShapeMeasurement(
            e=np.full((3, 2), value),
            dedg=np.full((3, 2, 2), value),
            flags=np.array([flag, False, False]),
        )

    straight = {"shearnet": {"sim": {"plus": measurement(1.0, True),
                                     "minus": measurement(3.0, False)},
                             # measured at one station only -> dropped, not
                             # partly averaged
                             "metacal": {"plus": measurement(1.0, False)}}}
    rotated = {"shearnet": {"sim": {"plus": measurement(5.0, False),
                                    "minus": measurement(1.0, False)},
                            "metacal": {}}}

    merged = harness._average_rotations([straight, rotated])
    assert set(merged["shearnet"]) == {"sim"}
    plus = merged["shearnet"]["sim"]["plus"]
    assert np.allclose(plus.e, 3.0) and np.allclose(plus.dedg, 3.0)
    assert plus.flags.tolist() == [True, False, False]
    assert np.allclose(merged["shearnet"]["sim"]["minus"].e, 2.0)


def test_shape_noise_cancellation_writes_the_twin_and_stays_reproducible(trained_run):
    """The rotated twin reaches the file, and SUMMARY still divides what is in it.

    Every number in SUMMARY has to be recomputable from the tables -- that is
    the standing contract of this file -- so turning on shape-noise cancellation
    has to carry the twin's columns, not just quietly average them away.
    """
    from astropy.io import fits

    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section["shape_noise_cancel"] = 4
    section["component"] = 0
    try:
        result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
        path = Path(benchmark.get("paths.root")) / section["output"]
        shear = benchmark.get("eval.evaluate.shear_true")
        with fits.open(path) as hdul:
            assert hdul["PRIMARY"].header["SNC"] == 4
            assert hdul["PRIMARY"].header["RING"] == "0,45,90,135"
            plus, minus = hdul["TAB_P"].data, hdul["TAB_M"].data
            summary = hdul["SUMMARY"].data
            for column in ("e_ngmix", "e_shearnet", "R_ngmix_sim", "R_shearnet_sim"):
                for suffix in ("_r45", "_r90", "_r135"):
                    assert f"{column}{suffix}" in plus.columns.names, column + suffix
                    assert f"{column}{suffix}" in minus.columns.names, column + suffix
            # each station is a different measurement of the same objects
            assert not np.allclose(plus["e_shearnet"], plus["e_shearnet_r90"])
            assert not np.allclose(plus["e_shearnet_r45"], plus["e_shearnet_r90"])
            # ...and the nuisance columns are NOT duplicated
            assert "s2n_r90" not in plus.columns.names

            def ring(table, name):
                return np.mean(
                    [np.asarray(table[f"{name}{s}"], dtype=float)
                     for s in ("", "_r45", "_r90", "_r135")],
                    axis=0,
                )

            for estimator in ("ngmix", "shearnet"):
                e_p, e_m = ring(plus, f"e_{estimator}"), ring(minus, f"e_{estimator}")
                keep = np.isfinite(e_p).all(axis=1) & np.isfinite(e_m).all(axis=1)
                by_hand = np.nanmean(0.5 * (e_p[keep, 0] - e_m[keep, 0])) / shear - 1.0
                row = summary[(summary["estimator"] == estimator)
                              & (summary["correction"] == "none")
                              & (summary["component"] == 0)]
                assert np.isclose(row["m"][0], by_hand, rtol=1e-10), estimator
    finally:
        section.clear()
        section.update(saved)
    assert result["shape_noise_cancel"] == 4


def test_the_leakage_pass_rings_too_and_cancels_the_intrinsic_shape(trained_run):
    """alpha is the shape-noise-limited number, so the ring matters most here.

    The LEAKAGE table has to carry every station AND the ring average under its
    own name, because that average -- not the unrotated station -- is what
    LEAKSUM reports and what the leakage panels should plot.
    """
    from astropy.io import fits

    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section["shape_noise_cancel"] = 2
    section["component"] = 0
    try:
        result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
        path = Path(benchmark.get("paths.root")) / section["output"]
        with fits.open(path) as hdul:
            leak = hdul["LEAKAGE"].data
            for estimator in ("ngmix", "shearnet"):
                station0 = np.asarray(leak[f"e_{estimator}_raw"], dtype=float)
                station90 = np.asarray(leak[f"e_{estimator}_raw_r90"], dtype=float)
                ring = np.asarray(leak[f"e_{estimator}_raw_ring"], dtype=float)
                # the ring column is exactly the mean of the stations
                np.testing.assert_allclose(ring, 0.5 * (station0 + station90))
                # ...and it is a genuinely different measurement
                assert not np.allclose(station0, station90)
                # the PSF response is ringed the same way
                assert f"Rpsf_{estimator}_sim_r90" in leak.columns.names
                assert f"Rpsf_{estimator}_sim_ring" in leak.columns.names
                # LEAKSUM reports the ring average, not station 0
                reported = np.asarray(
                    result[f"{estimator}_leakage_mean_e_raw"], dtype=float
                )
                np.testing.assert_allclose(
                    reported, np.nanmean(ring, axis=0), rtol=1e-10
                )
    finally:
        section.clear()
        section.update(saved)


def test_shape_noise_cancellation_is_off_by_default(trained_run):
    """It costs 2x the measurement, so it is never turned on behind your back."""
    benchmark, training = trained_run
    section = harness._section(benchmark, "evaluate")
    saved = dict(section)
    section.pop("shape_noise_cancel", None)
    section["component"] = 0
    try:
        result = harness._run_evaluation(benchmark, training, ("ngmix", "shearnet"))
        path = Path(benchmark.get("paths.root")) / section["output"]
        from astropy.io import fits

        with fits.open(path) as hdul:
            assert "e_shearnet_r90" not in hdul["TAB_P"].columns.names
            assert "e_shearnet_raw_ring" not in hdul["LEAKAGE"].columns.names
            assert hdul["PRIMARY"].header["SNC"] == 1
    finally:
        section.clear()
        section.update(saved)
    assert result["shape_noise_cancel"] == 1
