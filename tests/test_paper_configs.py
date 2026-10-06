"""The paper's four unit-test configs: what they hold each other to."""

from pathlib import Path

import pytest

from shearnet.config import Config
from shearnet.config.schema import flatten

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "configs" / "paper"
RUNGS = ("first", "second", "third", "fourth")


def _load(rung):
    return Config.from_file(PAPER / "unit_tests" / f"{rung}.yaml")


def _flat(rung):
    return flatten(_load(rung).to_dict())


def test_the_paper_has_exactly_the_four_unit_tests():
    yamls = sorted(p.relative_to(PAPER).as_posix() for p in PAPER.rglob("*.yaml"))
    assert yamls == [f"unit_tests/{r}.yaml" for r in sorted(RUNGS)]


@pytest.mark.parametrize("rung", RUNGS)
def test_each_unit_test_loads_and_lives_in_its_own_run_directory(rung):
    config = _load(rung)
    assert config.get("run_options.run_name") == f"d4_unit_{rung}"
    # relative to the config file, so the runs follow the clone wherever it lives
    assert config.get("run_options.outdir") == str(REPO / "runs" / "unit_tests" / rung)


@pytest.mark.parametrize("rung", RUNGS)
def test_ngmix_fits_the_psf_with_em5_like_litb_iii(rung):
    assert _load(rung).get("evaluation.ngmix.psf_model") == "em5"


def test_the_ladder_varies_only_the_simulation():
    """Table 1 is read down a column, which is valid only if the model is fixed.

    The one exception: UT1's circular PSF makes the PSF-orbit term a no-op
    (inloop.py refuses it), so it is switched off there and only there.
    """
    reference = _flat("fourth")
    for rung in RUNGS:
        flat = _flat(rung)
        differs = {k for k in reference if flat[k] != reference[k]
                   and not k.startswith(("simulation.", "run_options."))}
        expected = {"training.response.orbit_weight"} if rung == "first" else set()
        assert differs == expected, rung
    assert _flat("first")["training.response.orbit_weight"] == 0.0


def test_the_rungs_agree_with_the_paper_table():
    """UT1 ideal PSF; UT2 adds the PSFEx library; UT3 adds sizes; UT4 adds fluxes."""
    f = {r: _flat(r) for r in RUNGS}
    assert f["first"]["simulation.psf.mode"] == "ideal"
    for rung in ("second", "third", "fourth"):
        assert f[rung]["simulation.psf.mode"] == "superbit"
    assert [f[r]["simulation.hlr_type"] for r in RUNGS] == \
        ["constant", "constant", "catalog", "catalog"]
    assert [f[r]["simulation.flux_type"] for r in RUNGS] == \
        ["constant", "constant", "constant", "catalog"]


def test_a_misspelt_ngmix_psf_model_fails_at_load():
    from shearnet.config import ConfigError

    with pytest.raises(ConfigError, match="psf_model"):
        Config.from_dict({"schema_version": 1, "evaluation": {"ngmix": {"psf_model": "emm5"}}})
    for good in ("gauss", "em3", "em5", "coellip2"):
        Config.from_dict({"schema_version": 1, "evaluation": {"ngmix": {"psf_model": good}}})
