Quick reference
===============

Concise code snippets for the most common BLDFM workflows.  For step-by-step
tutorials, see the `Quickstart Tutorial <tutorial_quickstart.html>`_ and
`Full Tutorial <tutorial_full.html>`_.

Config-driven workflow (recommended)
-------------------------------------

The simplest way to run BLDFM is via a YAML configuration file.

**1. Create a config file** (``my_config.yaml``):

.. code-block:: yaml

    domain:
      nx: 256
      ny: 128
      xmax: 1000.0
      ymax: 500.0
      nz: 32
      modes: [256, 128]
      ref_lat: 50.95
      ref_lon: 11.586

    towers:
      - { name: tower_A, lat: 50.9505, lon: 11.5865, z_m: 10.0 }

    met:
      ustar: 0.4
      mol: -100.0
      wind_speed: 5.0
      wind_dir: 270.0

    solver:
      closure: MOST
      footprint: true

**2. Run from the command line:**

.. code-block:: bash

    $ bldfm run my_config.yaml
    $ bldfm run my_config.yaml --dry-run  # validate without running

**3. Or run from Python:**

.. code-block:: python

    from bldfm import load_config, run_bldfm_single, plot_footprint_field

    config = load_config("my_config.yaml")
    result = run_bldfm_single(config, config.towers[0])
    plot_footprint_field(result["flx"], result["grid"], contour_pcts=[0.5, 0.8])

Single-tower timeseries
-----------------------

The primary use case: one eddy-covariance tower with half-hourly forcing to
build a footprint climatology.

.. code-block:: python

    from bldfm import load_config, run_bldfm_timeseries
    from bldfm.plotting import plot_footprint_field, plot_footprint_timeseries
    import numpy as np

    config = load_config("examples/configs/timeseries.yaml")
    tower = config.towers[0]
    results = run_bldfm_timeseries(config, tower)

    # Time-averaged footprint
    mean_flx = np.mean([r["flx"] for r in results], axis=0)
    grid = results[0]["grid"]
    plot_footprint_field(mean_flx, grid, contour_pcts=[0.5, 0.7, 0.9])

    # Temporal evolution of source area
    plot_footprint_timeseries(results, grid, pcts=[0.5, 0.8])

Multi-tower timeseries
----------------------

For multiple towers and time-varying meteorology:

.. code-block:: python

    from bldfm import (
        load_config, run_bldfm_multitower,
        save_footprints_to_netcdf, plot_footprint_field,
    )

    config = load_config("my_config.yaml")
    results = run_bldfm_multitower(config)

    # Save to NetCDF
    save_footprints_to_netcdf(results, config, "output/footprints.nc")

    # Plot first tower, first timestep
    tower_name = config.towers[0].name
    result = results[tower_name][0]
    plot_footprint_field(result["flx"], result["grid"])

Parallel execution
------------------

For large runs, distribute work across CPU cores:

.. code-block:: python

    from bldfm import load_config, run_bldfm_parallel

    config = load_config("my_config.yaml")

    # Parallel over towers (each worker handles a full timeseries)
    results = run_bldfm_parallel(config, max_workers=4, parallel_over="towers")

    # Or parallel over both towers and timesteps
    results = run_bldfm_parallel(config, max_workers=8, parallel_over="both")

Synthetic data for testing
--------------------------

Generate reproducible test data without real observations:

.. code-block:: python

    from bldfm import generate_synthetic_timeseries, generate_towers_grid
    from bldfm.config_parser import parse_config_dict

    towers = generate_towers_grid(n_towers=4, z_m=10.0, layout="grid", seed=42)
    met = generate_synthetic_timeseries(n_timesteps=24, seed=42)

    config = parse_config_dict({
        "domain": {
            "nx": 256, "ny": 128, "xmax": 1000.0, "ymax": 500.0, "nz": 32,
            "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
        },
        "towers": towers,
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
    })

Low-level workflow
------------------

For full control, you can call the solver steps directly (as in the ``examples/low_level/minimal_example.py`` script):

.. code-block:: python

    from bldfm import compute_wind_fields, ideal_source, steady_state_transport_solver
    from bldfm.pbl_model import vertical_profiles

    # 1. Wind components
    u, v = compute_wind_fields(wind_speed=5.0, wind_dir=270.0)

    # 2. Vertical profiles
    z, profiles = vertical_profiles(n=32, meas_height=10.0, wind=(u, v), ustar=0.4)

    # 3. Surface flux
    srf_flx = ideal_source((256, 128), (1000.0, 500.0))

    # 4. Solve
    grid, conc, flx = steady_state_transport_solver(
        srf_flx, z, profiles, domain=(1000.0, 500.0), levels=32,
    )

Source area contours
--------------------

Visualize which spatial regions contribute a given fraction of the measured flux
using different contour geometries:

.. code-block:: python

    from bldfm import (
        get_source_area, source_area_circular, source_area_sector,
    )
    from bldfm.plotting import plot_source_area_contours, plot_source_area_gallery

    # After running the solver to get flx, grid, meas_pt, wind:

    # Single contour type
    g = source_area_circular(X, Y, meas_pt)
    rescaled = get_source_area(flx, g)
    plot_source_area_contours(flx, grid, rescaled, title="Circular contours")

    # Gallery of all 5 types
    fig, axes = plot_source_area_gallery(flx, grid, meas_pt, wind)

Five base functions are available: ``source_area_contribution`` (isopleth),
``source_area_circular`` (concentric circles), ``source_area_upwind`` (distance bands),
``source_area_crosswind`` (ridges), and ``source_area_sector`` (angular sectors).

See ``runs/low_level/source_area_example.py`` for a complete working example.

Reproducing manuscript figures
------------------------------

The ``runs/manuscript/`` directory contains the scripts that generate the figures
in the BLDFM paper.  To regenerate all of them at once (from the repo root):

.. code-block:: bash

    $ python runs/manuscript/generate_all.py

Individual figures can also be generated separately:

.. code-block:: bash

    $ python runs/manuscript/interface/comparison_footprint_unstable.py
    $ python runs/manuscript/interface/comparison_analytic.py

Outputs are saved to ``plots/``.

Testing the documentation build
-------------------------------

To build and verify the Sphinx documentation locally:

.. code-block:: bash

    cd docs

    # Standard build
    make html

    # CI-style build (treats warnings as errors)
    sphinx-build -W -b html source build/html

    # Preview in browser
    python -m http.server 8000 -d build/html

The ``-W`` flag causes any Sphinx warning to fail the build, which is what you want in CI to catch broken cross-references, missing modules, and formatting issues.
