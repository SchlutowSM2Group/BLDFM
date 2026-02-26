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

Pipeline architecture
---------------------

The solver pipeline proceeds in four stages. The high-level interface
(:py:mod:`bldfm.interface`) runs all four automatically; the low-level API lets
you call each stage individually.

.. code-block:: text

                      YAML file / dict
                            │
                   load_config / parse_config_dict
                            │
                            ▼
                       BLDFMConfig
                            │
            ┌───────────────┼───────────────┐
            │          High-level            │
            │   run_bldfm_single/timeseries  │
            │   run_bldfm_multitower         │
            │   run_bldfm_parallel           │
            └───────────────┬───────────────┘
                            │  internally calls ▼
              ┌─────────────┼─────────────────┐
              │                               │
              ▼                               │
     compute_wind_fields()  [utils]           │
              │                               │
              ▼                               │
     vertical_profiles()    [pbl_model]       │
              │                               │
              ▼                               │
     ideal_source()         [utils]           │
              │             (if no flux given) │
              ▼                               │
     steady_state_transport_solver()  [solver]│
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   result dict {grid, conc, flx, …}
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        GreensFunctionCache  save to     plot_*()
            [cache]         NetCDF     [plotting]
                            [io]

Design philosophy: high-level vs. low-level API
------------------------------------------------

BLDFM exposes two tiers of API, each targeting a different use case.

**High-level API** — :py:mod:`bldfm.interface`
    The four ``run_bldfm_*`` functions accept a :py:class:`~bldfm.config_parser.BLDFMConfig`
    object and return result dictionaries. Configuration objects carry every
    parameter (domain, towers, meteorology, solver options), making runs
    reproducible from a single YAML file. Use this tier for production science
    workflows, batch runs, and the CLI.

**Low-level API** — individual modules
    The core functions :py:func:`~bldfm.pbl_model.vertical_profiles`,
    :py:func:`~bldfm.solver.steady_state_transport_solver`,
    :py:func:`~bldfm.utils.compute_wind_fields`, and
    :py:func:`~bldfm.utils.ideal_source` can be called directly with plain
    NumPy arrays and scalars. This gives full control over intermediate results
    and is suited for custom parameter sweeps, debugging, or building new
    wrappers. Note that :py:func:`~bldfm.pbl_model.vertical_profiles` is
    intentionally not re-exported in the top-level ``__all__``; import it
    explicitly from :py:mod:`bldfm.pbl_model` when working at this level.

**Choosing a tier.** Start with the high-level API. Drop to the low-level API
when you need to inspect or modify intermediate quantities (e.g. vertical
profiles), run the solver with non-standard inputs, or integrate BLDFM into a
larger modelling framework.

API reference
-------------

For detailed information on functions and usage, refer to the :ref:`API Documentation <src>`.
