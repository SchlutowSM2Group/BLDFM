Full tutorial
=============

This tutorial covers every major BLDFM capability, organized by what you want
to accomplish.  It assumes you have completed the :doc:`tutorial_quickstart`
and have BLDFM installed.

.. contents:: Sections
   :local:
   :depth: 2

Prerequisites
-------------

.. code-block:: bash

    $ pip install -e ".[dev,plotting]"

All code below assumes you are in the repository root and have an active Python
session.  Small grid sizes are used throughout for fast execution.

.. code-block:: python

    import bldfm
    bldfm.initialize()


Configuration
-------------

BLDFM configurations define the computational domain, tower positions,
meteorological forcing, and solver settings.  You can load configs from YAML
files or build them programmatically in Python.

Loading a YAML config
^^^^^^^^^^^^^^^^^^^^^

BLDFM ships with several example configs in ``examples/configs/``.
``load_config()`` returns a ``BLDFMConfig`` dataclass:

.. code-block:: python

    config = bldfm.load_config("examples/configs/footprint.yaml")

    print(f"Domain: {config.domain.nx}x{config.domain.ny}, nz={config.domain.nz}")
    print(f"Tower: {config.towers[0].name}, z_m={config.towers[0].z_m}")
    print(f"Solver: closure={config.solver.closure}, footprint={config.solver.footprint}")

Building a config in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For programmatic workflows -- parameter sweeps, scripted experiments -- use
``parse_config_dict()`` with a plain Python dictionary that mirrors the YAML
structure.  When ``ref_lat``/``ref_lon`` are set, each tower's local ``(x, y)``
coordinates are computed automatically from its ``(lat, lon)`` via an
equirectangular projection:

.. code-block:: python

    from bldfm import parse_config_dict

    config = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [{"name": "tower_A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0}],
        "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
        "solver": {"closure": "MOST", "footprint": True},
    })

    print(f"Tower local coords: x={config.towers[0].x:.1f} m, y={config.towers[0].y:.1f} m")

Config schema reference
^^^^^^^^^^^^^^^^^^^^^^^

A ``BLDFMConfig`` contains six sub-configs:

**DomainConfig** -- computational grid geometry.

- ``nx``, ``ny``: grid points in x (cross-wind) and y (along-wind)
- ``nz``: number of vertical levels
- ``xmax``, ``ymax``: domain extent in metres
- ``modes``: Fourier modes ``[kx, ky]``; higher values improve accuracy
- ``halo``: zero-padding width (metres) to reduce spectral leakage
- ``ref_lat``, ``ref_lon``: reference origin for lat/lon to local coordinate conversion

**TowerConfig** -- measurement tower location.

- ``name``, ``lat``, ``lon``, ``z_m``: tower identity, coordinates, and height
- ``x``, ``y``: local coordinates (auto-computed from ``ref_lat``/``ref_lon``)

**MetConfig** -- meteorological forcing.  Use scalars for a single timestep
or lists of equal length for a timeseries.

- ``ustar``: friction velocity (m/s).  Provide *either* ``ustar`` or ``z0``,
  not both.  When ``z0`` (roughness length) is given, the model derives
  ``ustar`` internally via the PBL closure.
- ``mol``: Monin-Obukhov length (m).  Negative = unstable (daytime convection),
  positive = stable (nighttime), very large (~1e9) = neutral.
- ``wind_speed``, ``wind_dir``: horizontal wind speed (m/s) and direction
  (degrees; 0 = N, 90 = E, 180 = S, 270 = W).
- ``timestamps``: optional list of timestamp labels for each timestep.

**SolverConfig** -- numerical solver settings.

- ``closure``: PBL closure scheme -- ``MOST``, ``MOSTM``, ``CONSTANT``, or ``OAAHOC``.
- ``footprint``: ``true`` computes a flux footprint (Green's function at
  measurement height); ``false`` computes a concentration field from a
  surface source.
- ``precision``: ``single`` or ``double`` floating-point arithmetic.
- ``analytic``: if ``true``, uses the Kormann-Meixner analytical footprint
  model for comparison.

**OutputConfig** -- output format and directory.

**ParallelConfig** -- parallelism settings.

- ``num_threads``: BLAS/numba threads per worker.
- ``max_workers``: number of parallel worker processes.
- ``use_cache``: enable disk-based solver result caching.


Running simulations
-------------------

BLDFM provides four functions at progressive levels of complexity, from
a single solve to fully parallel multi-tower timeseries.

Single solve
^^^^^^^^^^^^

``run_bldfm_single()`` is the fundamental building block.  It takes a config
and a tower, runs one timestep, and returns a result dictionary:

.. code-block:: python

    from bldfm import run_bldfm_single

    result = run_bldfm_single(config, config.towers[0])

    print(f"Keys: {sorted(result.keys())}")
    print(f"Footprint shape: {result['flx'].shape}")  # (ny, nx)
    print(f"Met used: wind_speed={result['params']['wind_speed']}")

The result dict contains:

- ``grid``: tuple of ``(X, Y, Z)`` coordinate arrays
- ``flx``: 2D flux footprint field, shape ``(ny, nx)``
- ``conc``: 2D concentration field
- ``tower_name``, ``tower_xy``: tower identity and local coordinates
- ``timestamp``: timestep label or index
- ``params``: dict of meteorological parameters used for this solve

Timeseries
^^^^^^^^^^

For time-varying meteorology, provide lists in the met config.
``run_bldfm_timeseries()`` loops over all timesteps for a single tower and
returns a list of result dicts.  Use ``generate_synthetic_timeseries()`` to
create reproducible test data:

.. code-block:: python

    from bldfm import run_bldfm_timeseries, generate_synthetic_timeseries

    met = generate_synthetic_timeseries(n_timesteps=3, seed=42)
    config_ts = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [{"name": "tower_A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0}],
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
    })

    results = run_bldfm_timeseries(config_ts, config_ts.towers[0])
    print(f"Timesteps computed: {len(results)}")
    for r in results:
        print(f"  t={r['timestamp']}: max flux = {r['flx'].max():.6f}")

Multi-tower
^^^^^^^^^^^

``run_bldfm_multitower()`` runs all towers across all timesteps, returning a
dict of ``{tower_name: [result, ...]}``.  Use ``generate_towers_grid()`` to
create synthetic tower layouts (``"grid"``, ``"transect"``, or ``"random"``):

.. code-block:: python

    from bldfm import run_bldfm_multitower, generate_towers_grid

    towers = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
    config_mt = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
        },
        "towers": towers,
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
    })

    all_results = run_bldfm_multitower(config_mt)
    for name, res_list in all_results.items():
        print(f"{name}: {len(res_list)} timesteps")

Parallel execution
^^^^^^^^^^^^^^^^^^

For large runs, ``run_bldfm_parallel()`` distributes work across CPU cores
using ``ProcessPoolExecutor``.  Three strategies are available:

- ``"towers"``: each worker handles one tower's full timeseries
- ``"time"``: distribute timesteps across workers (per tower)
- ``"both"``: flatten all tower x timestep pairs across workers

Each subprocess automatically sets ``NUMBA_NUM_THREADS=1`` to avoid CPU
oversubscription.  The return format is identical to ``run_bldfm_multitower()``:

.. code-block:: python

    from bldfm import run_bldfm_parallel

    par_results = run_bldfm_parallel(config_mt, max_workers=2, parallel_over="towers")


Working with results
--------------------

Solver results are plain Python dictionaries containing NumPy arrays.  This
section covers common post-processing tasks.

Percentile contours
^^^^^^^^^^^^^^^^^^^

``extract_percentile_contour()`` computes the contour level that encloses a
given fraction of the cumulative footprint, along with the corresponding
area in square metres.  This is useful for quantifying the spatial extent of
the source area (e.g., "the 80 % source area covers 5000 m\ :sup:`2`\ "):

.. code-block:: python

    from bldfm import extract_percentile_contour

    level_50, area_50 = extract_percentile_contour(result["flx"], result["grid"], pct=0.5)
    level_80, area_80 = extract_percentile_contour(result["flx"], result["grid"], pct=0.8)

    print(f"50% contour: level={level_50:.6f}, area={area_50:.0f} m^2")
    print(f"80% contour: level={level_80:.6f}, area={area_80:.0f} m^2")

The 80 % area will always be larger than the 50 % area (it encloses more of
the footprint), while the 50 % contour level will be higher (it represents a
more concentrated core).

Source area analysis
^^^^^^^^^^^^^^^^^^^^

BLDFM provides five base functions for characterizing which spatial regions
contribute to the measured flux.  Each constructs a weighting surface ``g``
that, combined with the footprint via ``get_source_area(flx, g)``, produces
contours of a specific geometric type:

- ``source_area_contribution``: isopleth contours (standard footprint levels)
- ``source_area_circular``: concentric circles centred on the tower
- ``source_area_upwind``: upwind distance bands
- ``source_area_crosswind``: crosswind ridge contours
- ``source_area_sector``: angular sectors from the upwind axis

.. code-block:: python

    from bldfm import get_source_area, source_area_circular

    X, Y, Z = result["grid"]
    meas_pt = (config.towers[0].x, config.towers[0].y)

    g = source_area_circular(X, Y, meas_pt)
    rescaled = get_source_area(result["flx"], g)

See ``runs/low_level/source_area_example.py`` for a complete working example
with all five contour types.


I/O and caching
---------------

NetCDF export and import
^^^^^^^^^^^^^^^^^^^^^^^^

Save multi-tower results to CF-1.8 compliant NetCDF files and load them back
as an ``xr.Dataset``.  The file includes footprint and concentration fields,
meteorological metadata, tower coordinates, and global attributes:

.. code-block:: python

    from bldfm import save_footprints_to_netcdf, load_footprints_from_netcdf

    save_footprints_to_netcdf(all_results, config_mt, "output/footprints.nc")

    ds = load_footprints_from_netcdf("output/footprints.nc")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"CF Convention: {ds.attrs['Conventions']}")
    ds.close()

Green's function caching
^^^^^^^^^^^^^^^^^^^^^^^^^

In footprint mode, the solver output (Green's function) depends only on
vertical profiles and domain geometry -- not on the surface flux.  The
``GreensFunctionCache`` stores results as SHA-256-keyed ``.npz`` files so
that identical configurations skip the solver entirely:

.. code-block:: python

    from bldfm import GreensFunctionCache, run_bldfm_single

    cache = GreensFunctionCache(cache_dir=".bldfm_cache")

    result1 = run_bldfm_single(config, config.towers[0], cache=cache)  # cache miss
    result2 = run_bldfm_single(config, config.towers[0], cache=cache)  # cache hit (fast)

    cache.clear()  # clean up

Caching can also be enabled globally via ``parallel.use_cache: true`` in the
YAML config.


Visualization
-------------

BLDFM includes plotting functions for footprint fields, geospatial overlays,
wind roses, time series, source area contours, diagnostics, and interactive
HTML plots.  Core plots require only matplotlib; optional dependencies are
imported lazily and produce helpful messages when missing.

.. code-block:: python

    import matplotlib
    matplotlib.use("Agg")          # remove for interactive display
    import matplotlib.pyplot as plt

Footprint field
^^^^^^^^^^^^^^^

``plot_footprint_field()`` renders a 2D pcolormesh with optional percentile
contour overlays.  It works for both footprint and concentration fields:

.. code-block:: python

    from bldfm import plot_footprint_field

    ax = plot_footprint_field(
        result["flx"], result["grid"],
        contour_pcts=[0.5, 0.8],
        title="Footprint with 50% and 80% contours",
    )
    plt.savefig("plots/tutorial_contours.png", dpi=150, bbox_inches="tight")
    plt.close()

Map overlay (requires contextily)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``plot_footprint_on_map()`` overlays footprint contours and tower markers on
web map tiles.  Set ``land_cover=True`` to use ESA WorldCover 2021 instead of
street maps (requires owslib):

.. code-block:: python

    try:
        from bldfm import plot_footprint_on_map

        ax = plot_footprint_on_map(
            result["flx"], result["grid"], config,
            tower=config.towers[0],
            contour_pcts=[0.5, 0.8],
        )
        plt.savefig("plots/tutorial_map.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("Install contextily for map overlays: pip install contextily")

Wind rose (requires windrose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``plot_wind_rose()`` creates a polar wind rose from wind speed and direction
arrays:

.. code-block:: python

    try:
        from bldfm import plot_wind_rose

        ax = plot_wind_rose(
            met["wind_speed"], met["wind_dir"],
            title="Synthetic wind rose",
        )
        plt.savefig("plots/tutorial_windrose.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("Install windrose: pip install windrose")

Footprint timeseries
^^^^^^^^^^^^^^^^^^^^^

``plot_footprint_timeseries()`` tracks how footprint extent (area at given
percentiles) changes across timesteps:

.. code-block:: python

    from bldfm import plot_footprint_timeseries

    ax = plot_footprint_timeseries(
        results, results[0]["grid"],
        pcts=[0.5, 0.8],
        title="Footprint area evolution",
    )
    plt.savefig("plots/tutorial_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

Source area contour plots
^^^^^^^^^^^^^^^^^^^^^^^^^

``plot_source_area_gallery()`` produces a multi-panel figure showing all five
source area base function types side by side:

.. code-block:: python

    from bldfm.plotting import plot_source_area_gallery
    from bldfm import compute_wind_fields

    meas_pt = (config.towers[0].x, config.towers[0].y)
    wind = compute_wind_fields(5.0, 270.0)

    fig, axes = plot_source_area_gallery(result["flx"], result["grid"], meas_pt, wind)
    fig.savefig("plots/tutorial_source_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

Interactive plot (requires plotly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``plot_footprint_interactive()`` creates a Plotly figure that can be saved as
HTML for exploration in a browser:

.. code-block:: python

    try:
        from bldfm import plot_footprint_interactive

        fig = plot_footprint_interactive(result["flx"], result["grid"])
        fig.write_html("plots/tutorial_interactive.html")
    except ImportError:
        print("Install plotly: pip install plotly")

Diagnostic plots
^^^^^^^^^^^^^^^^

Three functions are available for inspecting solver behaviour and vertical
structure: ``plot_convergence()``, ``plot_vertical_profiles()``, and
``plot_vertical_slice()``.  See the `API Reference <src.html>`_ for full
signatures and usage.


Low-level API
-------------

For full control over every step of the simulation, you can call the solver
pipeline directly.  The high-level ``run_bldfm_single()`` wraps three steps:

1. ``compute_wind_fields(speed, direction)`` -- decompose wind into ``(u, v)``
   components.
2. ``vertical_profiles(nz, z_m, wind, ustar)`` -- compute vertical profiles of
   wind and eddy diffusivity.  Returns ``(z, profiles)`` where ``profiles`` is
   a 5-tuple ``(u, v, Kx, Ky, Kz)``.
3. ``steady_state_transport_solver(srf_flx, z, profiles, domain, levels, ...)``
   -- solve the advection-diffusion equation.

.. code-block:: python

    from bldfm import compute_wind_fields, ideal_source, steady_state_transport_solver
    from bldfm.pbl_model import vertical_profiles

    # Step 1: wind components
    u, v = compute_wind_fields(wind_speed=5.0, wind_dir=270.0)

    # Step 2: vertical profiles
    z, profiles = vertical_profiles(n=32, meas_height=10.0, wind=(u, v), ustar=0.4)
    u_prof, v_prof, Kx, Ky, Kz = profiles

    # Step 3: surface flux (or use np.zeros for footprint mode)
    srf_flx = ideal_source((256, 128), (1000.0, 500.0))

    # Step 4: solve
    grid, conc, flx = steady_state_transport_solver(
        srf_flx, z, profiles, domain=(1000.0, 500.0), levels=32,
    )

See the ``examples/low_level/`` directory for complete scripts:
``minimal_example.py``, ``footprint_example.py``, ``plot_profiles.py``,
and more.  For source area analysis, see ``runs/low_level/source_area_example.py``.


Command-line interface
----------------------

The ``bldfm`` CLI wraps the full workflow and calls ``initialize()``
automatically:

.. code-block:: bash

    # Validate a config without running the solver
    $ bldfm run examples/configs/footprint.yaml --dry-run

    # Run all towers and timesteps
    $ bldfm run examples/configs/multitower.yaml

    # Run and save footprint plots to plots/
    $ bldfm run examples/configs/multitower.yaml --plot


Further resources
-----------------

- `Quick Reference <reference.html>`_: concise code snippets for common
  workflows.
- `Example Scripts <runs.html>`_: ``examples/`` (config-driven),
  ``examples/low_level/`` (direct API), and ``runs/manuscript/`` (paper reproduction).
- `API Reference <src.html>`_: full function signatures and docstrings.
- Manuscript figures: ``python runs/manuscript/generate_all.py`` regenerates
  all paper figures.
