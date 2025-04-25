"""
This module contains unit tests for the `steady_state_transport_solver` function from the `solver.py` module. It validates the numerical solver against the analytical solution for the steady-state advection-diffusion equation.

The tests ensure that the numerical solution matches the analytical solution within a specified tolerance and that the outputs have the correct shapes and types.

Functions:
----------
- test_steady_state_transport_solver: Tests the solver against an analytical solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.solver import steady_state_transport_solver


def test_steady_state_transport_solver():
    """
    Tests the `steady_state_transport_solver` function against an analytical solution for the steady-state advection-diffusion equation with a point source at the center of the domain.

    The test validates:
        - The shape and type of the solver's outputs.
        - The numerical accuracy of the solver by comparing its results to the analytical solution within a specified tolerance.

    Raises:
    - AssertionError: If any of the output shapes, types, or numerical accuracy checks fail.
    """
    # Define test inputs
    nx, ny = 50, 50  # Grid size
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-50, 50, ny)
    z = np.linspace(0, 100, 1)  # Vertical grid points
    u = 1.0  # Constant wind speed
    K = 10.0  # Constant eddy diffusivity
    Q = 1.0  # Source strength
    profiles = (np.ones(len(z)) * u, np.zeros(len(z)), np.ones(len(z)) * K)
    domain = (100, 100)  # Domain size

    # Create a point source at the center of the domain
    srf_flx = np.zeros((ny, nx))
    srf_flx[ny // 4, nx // 4] = Q

    # Call the solver with analytic=True
    srf_conc_analytic, bg_conc_analytic, conc_analytic, flx_analytic = (
        steady_state_transport_solver(srf_flx, z, profiles, domain, analytic=True)
    )

    # Call the solver with analytic=False
    srf_conc_numeric, bg_conc_numeric, conc_numeric, flx_numeric = (
        steady_state_transport_solver(srf_flx, z, profiles, domain, analytic=False)
    )

    # Assertions to validate numeric output shapes
    assert srf_conc_numeric.shape == (ny, nx), "Surface concentration shape mismatch"
    assert isinstance(
        bg_conc_numeric, float
    ), "Background concentration should be a float"
    assert conc_numeric.shape == (ny, nx), "Concentration shape mismatch"
    assert flx_numeric.shape == (ny, nx), "Flux shape mismatch"

    # Compare the results
    assert np.allclose(
        srf_conc_analytic, srf_conc_numeric, atol=1e-2
    ), "Surface concentration mismatch between analytic and numeric"
    assert np.isclose(
        bg_conc_analytic, bg_conc_numeric, atol=1e-2
    ), "Background concentration mismatch between analytic and numeric"
    assert np.allclose(
        conc_analytic, conc_numeric, atol=1e-2
    ), "Concentration mismatch between analytic and numeric"
    assert np.allclose(
        flx_analytic, flx_numeric, atol=1e-2
    ), "Flux mismatch between analytic and numeric"

    # Plot solutions
    plt.figure(figsize=(12, 6))

    # Plot numerical solution
    plt.subplot(1, 2, 1)
    plt.title("Numerical Solution")
    plt.imshow(
        srf_conc_numeric, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()]
    )
    plt.colorbar()

    # Plot analytical solution
    plt.subplot(1, 2, 2)
    plt.title("Analytical Solution")
    plt.imshow(
        srf_conc_analytic, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()]
    )
    plt.colorbar()
    plt.savefig("plots/analytical_vs_numerical_solution.png")

    # Compare numerical and analytical solutions
    error = np.abs(srf_conc_numeric - srf_conc_analytic)
    max_error = np.max(error)
    assert (
        max_error < 1e-2
    ), f"Numerical solution deviates too much from analytical solution (max error: {max_error})"


if __name__ == "__main__":
    # Run the test manually (optional, for debugging)
    test_steady_state_transport_solver()
