Package overview
================
This package provides a modular framework for modeling atmospheric dispersion processes, focusing on the planetary boundary layer (PBL) and scalar transport. It includes three core modules in `src` for computing vertical profiles, solving advection-diffusion equations, and providing utility functions for diagnostics and input generation. Example run scripts and a test suite are also included to demonstrate usage and ensure reliability.

Core modules in the ``src`` subpackage
--------------------------------------

1. :py:mod:`src.pbl_model`
    Computes vertical profiles of horizontal wind and eddy diffusivity in the planetary boundary layer using Monin-Obukhov Similarity Theory (MOST).

    **Features:**
        - Implements surface-layer closure via stability corrections.
        - Supports parameterizations of ``u_h(z)``, ``K_h(z)``, and ``K_z(z)``.
        - Easy to extend with more sophisticated turbulence closures.

    **Key functions:**
        - :py:func:`src.pbl_model.vertical_profiles`
        - :py:func:`src.pbl_model.psi`
        - :py:func:`src.pbl_model.phi`

2. :py:mod:`src.solver`
    Solves the steady-state advection-diffusion equation for scalar concentration fields using FFT-based methods and numerical integration schemes.

    **Features:**
       - Supports Fourier-transformed formulation for horizontal advection-diffusion.
       - Uses linear shooting for boundary value problems in the vertical.
       - Implements Green's function solver for footprint modeling.

    **Key functions:**
       - :py:func:`src.solver.steady_state_transport_solver`
       - :py:func:`src.solver.ivp_solver`

3. :py:mod:`src.utils`
    Provides utility functions for preprocessing, synthetic input generation, and diagnostics.

    **Features:**
       - Source field generation (e.g., circular/point sources).
       - Wind field constructors and FFT utilities.
       - Pointwise convolution for footprint validation.

    **Key functions:**
       - :py:func:`src.utils.compute_wind_fields`
       - :py:func:`src.utils.point_source`
       - :py:func:`src.utils.ideal_source`
       - :py:func:`src.utils.point_measurement`

Additional subpackages
----------------------

1. ``runs``
    Provides example workflows and pre-configured scripts to demonstrate the usage of the package. These scripts showcase how to combine the core modules (`pbl_model`, `solver`, and `utils`) for practical applications, such as footprint modeling or dispersion analysis.

    **Features:**
       - Example configurations for common use cases.
       - Demonstrates integration of vertical profiles, transport solvers, and diagnostics.

2. ``tests``
    Contains the test suite for validating the functionality and accuracy of the package. The tests ensure that the core modules (`pbl_model`, `solver`, and `utils`) work as expected and provide a framework for extending test coverage.

    **Features:**
       - Unit tests for individual functions.
       - Integration tests for workflows combining multiple modules.
       - Easy-to-run test scripts for contributors.

Module interactions
-------------------

- ``pbl_model`` provides the vertical profiles required by the ``solver`` module.
- ``solver`` handles the core transport computation, optionally using Green's functions for convolution with surface fluxes.
- ``utils`` supports both ``pbl_model`` and ``solver`` with reusable tools for test generation and output diagnostics.
- ``runs`` contains ....
- ``tests`` ensures the reliability and correctness of the package.

API reference
-------------

For detailed information on functions and usage, refer to the :ref:`API Documentation <src>`.