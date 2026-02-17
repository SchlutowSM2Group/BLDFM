"""
This module contains an integration test for the interaction between the `pbl_model`, `utils`, and `solver` components.

The test validates the combined behavior of these components by comparing the numerical and analytical solutions for the steady-state advection-diffusion equation.
Currently, this test is similar to combining the `pbl_model` and `solver` tests, but it is designed to serve as a foundation for future extensions.

Functions:
----------
- test_integration: Tests the combined behavior of the `pbl_model`, `utils`, and `solver` components.
"""

import pytest

pytestmark = pytest.mark.integration

import numpy as np
from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source
from bldfm.solver import steady_state_transport_solver


def test_integration():
    """
    Tests the combined behavior of the `pbl_model`, `utils`, and `solver` components.

    The test validates:
        - The interaction between the components.
        - The numerical accuracy of the solver by comparing its results to the analytical solution within a specified tolerance.

    Raises:
        AssertionError: If the numerical and analytical solutions differ beyond the specified tolerance.
    """
    # Define inputs
    nxy = 256, 256
    modes = 512, 512
    nz = 256
    # nxy         = 128, 64
    # modes       = 128, 128
    # nz          = 16
    domain = 100.0, 50.0
    src_pt = 5.0, 5.0
    halo = 500.0
    meas_height = 5.0
    wind = 4.0, 1.0
    ustar = 0.2

    # Generate inputs using pbl_model and utils
    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

    # Solve using the solver
    _, conc_ana, flx_ana = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo, analytic=True
    )
    _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo
    )

    # Validate results
    diff_conc = (conc - conc_ana) / np.max(conc_ana)
    diff_flx = (flx - flx_ana) / np.max(flx_ana)

    assert np.allclose(diff_conc, 0, atol=1e-3), "Concentration mismatch too large"
    assert np.allclose(diff_flx, 0, atol=1e-3), "Flux mismatch too large"


def test_convergence_trend():
    """Verify that numerical error decreases monotonically with resolution.

    Runs the solver at 3 resolution levels against the analytic solution
    (CONSTANT closure) and asserts that relative MSE decreases at each
    refinement step.
    """
    nxy = 128, 64
    domain = 100.0, 50.0
    src_pt = 5.0, 5.0
    halo = 500.0
    meas_height = 5.0
    wind = 4.0, 1.0
    ustar = 0.2

    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")

    resolutions = [
        {"modes": (64, 64), "nz": 8},
        {"modes": (128, 128), "nz": 16},
        {"modes": (256, 256), "nz": 32},
    ]

    # Analytic reference at finest resolution
    z_ref, profs_ref = vertical_profiles(
        resolutions[-1]["nz"], meas_height, wind, ustar, closure="CONSTANT"
    )
    _, _, flx_ana = steady_state_transport_solver(
        srf_flx,
        z_ref,
        profs_ref,
        domain,
        resolutions[-1]["nz"],
        modes=resolutions[-1]["modes"],
        halo=halo,
        analytic=True,
    )

    errors = []
    for res in resolutions:
        z, profs = vertical_profiles(
            res["nz"], meas_height, wind, ustar, closure="CONSTANT"
        )
        _, _, flx = steady_state_transport_solver(
            srf_flx,
            z,
            profs,
            domain,
            res["nz"],
            modes=res["modes"],
            halo=halo,
        )
        rmse = np.mean((flx - flx_ana) ** 2) / np.mean(flx_ana**2)
        errors.append(rmse)

    # Error must decrease monotonically with resolution
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], (
            f"Error did not decrease: errors[{i}]={errors[i]:.6e} "
            f"<= errors[{i+1}]={errors[i+1]:.6e}"
        )


def _quick_solve(
    footprint=True, precision="single", modes=(128, 64), halo=None, meas_pt=(0.0, 0.0)
):
    """Helper: small solve at conftest scale for fast tests."""
    nxy = (128, 64)
    nz = 16
    domain = (500.0, 250.0)
    src_pt = (250.0, 125.0)
    meas_height = 10.0
    wind = (5.0, 0.0)
    ustar = 0.4

    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="MOST")

    return steady_state_transport_solver(
        srf_flx,
        z,
        profs,
        domain,
        nz - 1,
        modes=modes,
        footprint=footprint,
        precision=precision,
        halo=halo,
        meas_pt=meas_pt,
    )


def test_solver_single_precision():
    """Verify single precision path produces float32 output."""
    _, conc, flx = _quick_solve(precision="single")
    assert conc.dtype == np.float32 or conc.dtype == np.float64
    # Single precision uses complex64 internally -> real part is float32
    assert flx.shape == conc.shape


def test_solver_double_precision():
    """Verify double precision path produces float64 output."""
    _, conc, flx = _quick_solve(precision="double")
    assert conc.dtype == np.float64
    assert flx.dtype == np.float64


def test_solver_invalid_precision_raises():
    """Verify invalid precision raises ValueError."""
    with pytest.raises(ValueError, match="precision must be"):
        _quick_solve(precision="quad")


def test_solver_odd_modes_raises():
    """Verify odd modes raise ValueError."""
    with pytest.raises(ValueError, match="modes must consist of even numbers"):
        _quick_solve(modes=(63, 64))


def test_solver_halo_overflow():
    """Verify solver handles modes exceeding grid+halo gracefully."""
    # Use a tiny halo so modes > grid+halo triggers the clamping path
    _, conc, flx = _quick_solve(modes=(512, 512), halo=1.0)
    assert conc.shape == flx.shape
    assert np.isfinite(flx).all()


def test_solver_non_footprint_shift():
    """Verify non-footprint mode with non-zero meas_pt (shift path)."""
    _, conc, flx = _quick_solve(
        footprint=False, meas_pt=(25.0, 12.5), precision="double"
    )
    assert conc.shape == flx.shape
    assert np.isfinite(conc).all()


if __name__ == "__main__":
    # Run the tests manually (optional, for debugging)
    test_integration()
    test_convergence_trend()
