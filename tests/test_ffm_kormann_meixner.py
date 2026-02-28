"""Unit tests for the Kormann & Meixner (2001) footprint model."""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from bldfm.ffm_kormann_meixner import (
    _mParam,
    _nParam,
    _phiC,
    _phiM,
    _psiM,
    estimateFootprint,
    estimateZ0,
    von_karman,
)


# ---------------------------------------------------------------------------
# Stability function tests (private helpers)
# ---------------------------------------------------------------------------
# Note: L=0 causes division-by-zero in the stable branch, so we use L=1e6
# for near-neutral conditions.


@pytest.mark.parametrize(
    "zm, mo_len, expected",
    [
        # Near-neutral (large positive L): phi_m = 1 + 5*10/1e6
        (np.array([10.0]), np.array([1e6]), np.array([1.00005])),
        # Unstable: phi_m = (1 - 16*10/(-100))^(-0.25) = 2.6^(-0.25)
        (np.array([10.0]), np.array([-100.0]), np.array([2.6 ** (-0.25)])),
        # Stable: phi_m = 1 + 5*10/100 = 1.5
        (np.array([10.0]), np.array([100.0]), np.array([1.5])),
    ],
)
def test_phiM_stability_conditions(zm, mo_len, expected):
    """Verify Eq. (33) for near-neutral, unstable, and stable conditions."""
    result = _phiM(zm, mo_len)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "zm, mo_len, expected",
    [
        # Near-neutral: phi_c = 1 + 5*10/1e6
        (np.array([10.0]), np.array([1e6]), np.array([1.00005])),
        # Unstable: phi_c = (1 - 16*10/(-100))^(-0.5) = 2.6^(-0.5)
        (np.array([10.0]), np.array([-100.0]), np.array([2.6 ** (-0.5)])),
        # Stable: phi_c = 1 + 5*10/100 = 1.5
        (np.array([10.0]), np.array([100.0]), np.array([1.5])),
    ],
)
def test_phiC_stability_conditions(zm, mo_len, expected):
    """Verify Eq. (34) for near-neutral, unstable, and stable conditions."""
    result = _phiC(zm, mo_len)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "zm, mo_len, expected",
    [
        # Near-neutral (stable branch): psi_m = 5*10/1e6
        (np.array([10.0]), np.array([1e6]), np.array([5 * 10.0 / 1e6])),
        # Stable: psi_m = 5*10/100 = 0.5
        (np.array([10.0]), np.array([100.0]), np.array([0.5])),
    ],
)
def test_psiM_stability_conditions(zm, mo_len, expected):
    """Verify Eq. (35) for near-neutral and stable conditions."""
    result = _psiM(zm, mo_len)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_psiM_unstable():
    """Verify Eq. (35) for unstable conditions via explicit formula."""
    zm = np.array([10.0])
    mo_len = np.array([-100.0])
    inv_phi_m = (1 - 16 * 10.0 / (-100.0)) ** 0.25  # 2.6^0.25
    expected = (
        -2 * np.log(0.5 * (1 + inv_phi_m))
        - np.log(0.5 * (1 + inv_phi_m**2))
        + 2 * np.arctan(inv_phi_m)
        - np.pi * 0.5
    )
    result = _psiM(zm, mo_len)
    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-10)


@pytest.mark.parametrize(
    "zm, ws, ustar, mo_len, expected",
    [
        # Near-neutral: m = ustar * phi_m / (k * ws) where phi_m ~ 1
        # m = 0.4 * 1.00005 / (0.4 * 4.0) = 0.250013
        (
            np.array([10.0]),
            np.array([4.0]),
            np.array([0.4]),
            np.array([1e6]),
            np.array([0.4 * (1 + 5 * 10.0 / 1e6) / (von_karman * 4.0)]),
        ),
        # Unstable: m = 0.4 * 2.6^(-0.25) / (0.4 * 4.0)
        (
            np.array([10.0]),
            np.array([4.0]),
            np.array([0.4]),
            np.array([-100.0]),
            np.array([0.4 * 2.6 ** (-0.25) / (von_karman * 4.0)]),
        ),
    ],
)
def test_mParam_values(zm, ws, ustar, mo_len, expected):
    """Verify m parameter (Eq. 36) for near-neutral and unstable."""
    result = _mParam(zm, ws, ustar, mo_len)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "zm, mo_len, expected",
    [
        # Near-neutral: n = 1 / (1 + 5*10/1e6) ~ 0.99995
        (np.array([10.0]), np.array([1e6]), np.array([1 / (1 + 5 * 10.0 / 1e6)])),
        # Unstable: n = (1 - 24*10/(-100)) / (1 - 16*10/(-100)) = 3.4 / 2.6
        (np.array([10.0]), np.array([-100.0]), np.array([3.4 / 2.6])),
        # Stable: n = 1 / (1 + 5*10/100) = 1/1.5
        (np.array([10.0]), np.array([100.0]), np.array([1.0 / 1.5])),
    ],
)
def test_nParam_values(zm, mo_len, expected):
    """Verify n parameter (Eq. 36) for near-neutral, unstable, and stable."""
    result = _nParam(zm, mo_len)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_phiM_vectorized():
    """Verify element-wise branching with mixed stable/unstable array."""
    zm = np.array([10.0, 10.0])
    mo_len = np.array([-100.0, 100.0])
    result = _phiM(zm, mo_len)
    np.testing.assert_allclose(result[0], 2.6 ** (-0.25), rtol=1e-4)
    np.testing.assert_allclose(result[1], 1.5, rtol=1e-4)


# ---------------------------------------------------------------------------
# estimateFootprint tests
# ---------------------------------------------------------------------------

_FP_COMMON = dict(
    zm=10.0,
    z0=0.1,
    ws=4.0,
    ustar=0.4,
    mo_len=-100.0,
    sigma_v=0.3,
    grid_res=5.0,
    mxy=[0, 0],
)


def test_estimateFootprint_grid_shape():
    """Output grid dimensions must match grid_domain / grid_res."""
    gx, gy, gf = estimateFootprint(
        **_FP_COMMON,
        grid_domain=[-200, 200, -100, 100],
        wd=270.0,
    )
    # x: (200-(-200))/5 = 80 cols, y: (100-(-100))/5 = 40 rows
    assert gx.shape == (40, 80)
    assert gy.shape == (40, 80)
    assert gf.shape == (40, 80)


def test_estimateFootprint_non_negative():
    """All footprint values must be >= 0."""
    _, _, gf = estimateFootprint(
        **_FP_COMMON,
        grid_domain=[-200, 200, -100, 100],
        wd=270.0,
    )
    assert np.all(gf >= 0)


def test_estimateFootprint_mass_conservation():
    """Footprint should capture a substantial fraction of total flux."""
    _, _, gf = estimateFootprint(
        zm=10.0,
        z0=0.1,
        ws=4.0,
        ustar=0.4,
        mo_len=1e6,  # near-neutral for well-behaved footprint
        sigma_v=0.3,
        grid_domain=[-500, 3000, -1000, 1000],
        grid_res=5.0,
        mxy=[0, 0],
        wd=270.0,
    )
    total = np.sum(gf)
    assert 0.5 < total <= 1.05, f"Footprint sum {total:.4f} outside expected range"


def test_estimateFootprint_wind_rotation():
    """Peak location should change when wind direction rotates."""
    common = dict(
        zm=10.0,
        z0=0.1,
        ws=4.0,
        ustar=0.4,
        mo_len=1e6,
        sigma_v=0.3,
        grid_domain=[-200, 200, -200, 200],
        grid_res=5.0,
        mxy=[0, 0],
    )
    _, _, gf_north = estimateFootprint(**common, wd=0.0)
    _, _, gf_east = estimateFootprint(**common, wd=90.0)
    peak_north = np.unravel_index(np.argmax(gf_north), gf_north.shape)
    peak_east = np.unravel_index(np.argmax(gf_east), gf_east.shape)
    assert peak_north != peak_east


def test_estimateFootprint_negative_U_warns():
    """When z0 >> zm, U becomes negative and function returns zeros with warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, gf = estimateFootprint(
            zm=10.0,
            z0=100.0,  # z0 > zm: log(zm/z0) strongly negative -> U < 0
            ws=4.0,
            ustar=0.4,
            mo_len=-100.0,
            sigma_v=0.3,
            grid_domain=[-200, 200, -100, 100],
            grid_res=10.0,
            mxy=[0, 0],
            wd=270.0,
        )
        assert np.all(gf == 0.0)
        assert len(w) == 1


# ---------------------------------------------------------------------------
# estimateZ0 tests
# ---------------------------------------------------------------------------


def test_estimateZ0_input_mismatch_raises():
    """Mismatched array lengths must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="same length"):
        estimateZ0(
            zm=np.array([10.0, 10.0]),
            ws=np.array([4.0]),  # wrong length
            wd=np.array([270.0, 270.0]),
            ustar=np.array([0.4, 0.4]),
            mo_len=np.array([-100.0, -100.0]),
        )


def test_estimateZ0_no_smoothing():
    """With half_wd_win < 1, should return raw formula without smoothing."""
    zm = np.array([10.0])
    ws = np.array([4.0])
    wd = np.array([270.0])
    ustar_arr = np.array([0.4])
    mo_len_arr = np.array([-100.0])

    result = estimateZ0(zm, ws, wd, ustar_arr, mo_len_arr, half_wd_win=0)

    psi_m = _psiM(zm, mo_len_arr)
    expected = zm * np.exp(psi_m - von_karman * ws / ustar_arr)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
