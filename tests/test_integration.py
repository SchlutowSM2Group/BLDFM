"""
This module contains an integration test for the interaction between the `pbl_model`, `utils`, and `solver` components.

The test validates the combined behavior of these components by comparing the numerical and analytical solutions for the steady-state advection-diffusion equation.
Currently, this test is similar to combining the `pbl_model` and `solver` tests, but it is designed to serve as a foundation for future extensions.

Functions:
----------
- test_integration: Tests the combined behavior of the `pbl_model`, `utils`, and `solver` components.
"""

import numpy as np
from src.pbl_model import vertical_profiles
from src.utils import point_source
from src.solver import steady_state_transport_solver


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
    fetch = 500.0
    meas_height = 5.0
    wind = 4.0, 1.0
    ustar = 0.2

    # Generate inputs using pbl_model and utils
    srf_flx = point_source(nxy, domain, src_pt)
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

    # Solve using the solver
    _, _, conc_ana, flx_ana = steady_state_transport_solver(
        srf_flx, z, profs, domain, modes=modes, fetch=fetch, analytic=True
    )
    _, _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, modes=modes, fetch=fetch, ivp_method="TSEI3"
    )

    # Validate results
    diff_conc = (conc - conc_ana) / np.max(conc_ana)
    diff_flx = (flx - flx_ana) / np.max(flx_ana)

    assert np.allclose(diff_conc, 0, atol=1e-3), "Concentration mismatch too large"
    assert np.allclose(diff_flx, 0, atol=1e-3), "Flux mismatch too large"


if __name__ == "__main__":
    # Run the test manually (optional, for debugging)
    test_integration()
