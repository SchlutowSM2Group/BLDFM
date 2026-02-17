Package overview
================

BLDFM provides a modular framework for modelling atmospheric dispersion and flux footprints in the planetary boundary layer (PBL). It numerically solves the three-dimensional steady-state advection-diffusion equation and computes Green's function footprints for flux tower networks.

Core modules
------------

1. :py:mod:`bldfm.pbl_model`
    Computes vertical profiles of horizontal wind and eddy diffusivity using Monin-Obukhov Similarity Theory (MOST).

    **Key functions:**
        - :py:func:`bldfm.pbl_model.vertical_profiles`
        - :py:func:`bldfm.pbl_model.psi`
        - :py:func:`bldfm.pbl_model.phi`

2. :py:mod:`bldfm.solver`
    Solves the steady-state advection-diffusion equation using FFT-based methods with linear shooting for the vertical boundary value problem.

    **Key functions:**
        - :py:func:`bldfm.solver.steady_state_transport_solver`
        - :py:func:`bldfm.solver.ivp_solver`

3. :py:mod:`bldfm.utils`
    Utility functions for wind field construction, source generation, and diagnostics.

    **Key functions:**
        - :py:func:`bldfm.utils.compute_wind_fields`
        - :py:func:`bldfm.utils.ideal_source`
        - :py:func:`bldfm.utils.point_measurement`

Configuration and interface
---------------------------

4. :py:mod:`bldfm.config_parser`
    YAML-based configuration with dataclass schema. Defines ``BLDFMConfig``, ``TowerConfig``, ``DomainConfig``, ``MetConfig``, and related classes.

5. :py:mod:`bldfm.interface`
    High-level functions that encapsulate the full workflow:

    - :py:func:`~bldfm.interface.run_bldfm_single` -- single tower, single timestep
    - :py:func:`~bldfm.interface.run_bldfm_timeseries` -- single tower, all timesteps
    - :py:func:`~bldfm.interface.run_bldfm_multitower` -- all towers, all timesteps
    - :py:func:`~bldfm.interface.run_bldfm_parallel` -- parallel execution over towers, time, or both

6. :py:mod:`bldfm.cli`
    Command-line interface: ``bldfm run config.yaml [--dry-run]``.

Data and I/O
------------

7. :py:mod:`bldfm.synthetic`
    Generates synthetic meteorological timeseries and tower configurations for testing and prototyping.

8. :py:mod:`bldfm.io`
    NetCDF export and import of footprint results using xarray with CF-1.8 metadata.

9. :py:mod:`bldfm.cache`
    Disk-based cache for Green's function results using SHA-256 keyed ``.npz`` files.

Plotting
--------

10. :py:mod:`bldfm.plotting`
     Visualisation functions for footprint fields, geospatial map overlays, wind roses, and temporal footprint evolution.

Module interactions
-------------------

- :py:mod:`bldfm.pbl_model` provides the vertical profiles required by :py:mod:`bldfm.solver`.
- :py:mod:`bldfm.interface` orchestrates the full pipeline (wind fields, profiles, solver) driven by :py:mod:`bldfm.config_parser`.
- :py:mod:`bldfm.cache` accelerates repeated solves by caching Green's functions.
- :py:mod:`bldfm.io` saves and loads multi-tower, multi-timestep results.
- :py:mod:`bldfm.plotting` visualises solver outputs.

API reference
-------------

For detailed information on functions and usage, refer to the :ref:`API Documentation <src>`.
