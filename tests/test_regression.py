"""Regression tests: compare solver outputs against saved reference data.

Run with:
    pytest -m regression                          # check against references
    pytest --update-references -m regression      # regenerate references
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

REF_DIR = Path(__file__).parent / "references"

# Per-scenario tolerances (MOST closure, deterministic on same platform).
SCENARIOS = {
    "single_footprint": {"atol": 1e-6, "rtol": 1e-5},
    "source_area": {"atol": 1e-6, "rtol": 1e-5},
    "plume_3d": {"atol": 1e-6, "rtol": 1e-5},
    "timeseries_step0": {"atol": 1e-6, "rtol": 1e-5},
    "timeseries_step2": {"atol": 1e-6, "rtol": 1e-5},
}


def _save_reference(name, grid, conc, flx):
    """Save grid, conc, and flx arrays to references/{name}.npz."""
    REF_DIR.mkdir(exist_ok=True)
    X, Y, Z = grid
    np.savez_compressed(REF_DIR / f"{name}.npz", X=X, Y=Y, Z=Z, conc=conc, flx=flx)


def _load_reference(name):
    """Load reference .npz, raising FileNotFoundError if missing."""
    path = REF_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {path}\n"
            f"Run 'pytest --update-references -m regression' to generate it."
        )
    return dict(np.load(path))


def _compare(name, grid, conc, flx, update_refs):
    """Save-and-skip or load-and-assert against reference data."""
    if update_refs:
        _save_reference(name, grid, conc, flx)
        pytest.skip(f"Reference updated: {name}")

    ref = _load_reference(name)

    assert (
        conc.shape == ref["conc"].shape
    ), f"{name}: conc shape mismatch {conc.shape} vs {ref['conc'].shape}"
    assert (
        flx.shape == ref["flx"].shape
    ), f"{name}: flx shape mismatch {flx.shape} vs {ref['flx'].shape}"

    tols = SCENARIOS[name]
    np.testing.assert_allclose(
        conc,
        ref["conc"],
        atol=tols["atol"],
        rtol=tols["rtol"],
        err_msg=f"{name}: conc regression",
    )
    np.testing.assert_allclose(
        flx,
        ref["flx"],
        atol=tols["atol"],
        rtol=tols["rtol"],
        err_msg=f"{name}: flx regression",
    )


def test_regression_single_footprint(single_run_result, update_references):
    r = single_run_result
    _compare("single_footprint", r["grid"], r["conc"], r["flx"], update_references)


def test_regression_source_area(source_area_result_session, update_references):
    r = source_area_result_session
    _compare("source_area", r["grid"], r["conc"], r["flx"], update_references)


def test_regression_plume_3d(plume_3d_result_session, update_references):
    r = plume_3d_result_session
    _compare("plume_3d", r["grid"], r["conc"], r["flx"], update_references)


def test_regression_timeseries_step0(timeseries_results_session, update_references):
    r = timeseries_results_session[0]
    _compare("timeseries_step0", r["grid"], r["conc"], r["flx"], update_references)


def test_regression_timeseries_step2(timeseries_results_session, update_references):
    r = timeseries_results_session[2]
    _compare("timeseries_step2", r["grid"], r["conc"], r["flx"], update_references)
