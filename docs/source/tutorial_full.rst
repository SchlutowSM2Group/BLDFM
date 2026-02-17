Full tutorial: verifying all v1.0 features
==========================================

This tutorial walks through every feature added in the v1.0 expansion,
organised by implementation phase.  Each section includes verification checks
so you can confirm the feature works correctly.

.. contents:: Sections
   :local:
   :depth: 2

Prerequisites
-------------

.. code-block:: bash

    $ pip install -e ".[dev,plotting]"

All code below assumes you are in the repository root and have an active Python
session (script or interactive shell).  Small grid sizes are used throughout
for fast execution.

.. code-block:: python

    import bldfm
    bldfm.initialize()


Phase 0: Housekeeping
---------------------

Explicit initialisation
^^^^^^^^^^^^^^^^^^^^^^^

Import-time side effects were removed in Phase 0.  You must call
``initialize()`` explicitly before running simulations.

.. code-block:: python

    import bldfm

    # Before initialisation
    print(f"Before: bldfm._initialized = {bldfm._initialized}")

    bldfm.initialize()

    print(f"After: bldfm._initialized = {bldfm._initialized}")

    # Verify
    import os
    assert bldfm._initialized is True
    assert os.path.isdir("logs")
    assert os.path.isdir("plots")
    print("Phase 0 — initialize(): OK")

Calling ``initialize()`` a second time is a no-op.

Vertical profiles return tuple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``vertical_profiles`` now returns a **5-tuple** ``(u, v, Kx, Ky, Kz)`` as the
profiles component.

.. code-block:: python

    from bldfm.pbl_model import vertical_profiles

    z, profiles = vertical_profiles(n=8, meas_height=10.0, wind=(5.0, 0.0), ustar=0.4)
    u, v, Kx, Ky, Kz = profiles

    print(f"z shape: {z.shape}")
    print(f"u shape: {u.shape}")
    print(f"u at measurement height: {u[-1]:.3f}")

    assert len(profiles) == 5, "Expected 5-tuple: (u, v, Kx, Ky, Kz)"
    print("Phase 0 — vertical_profiles 5-tuple: OK")

plot_profiles.py compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``runs/low_level/plot_profiles.py`` script was updated to unpack the new tuple:

.. code-block:: bash

    $ python -m runs.low_level.plot_profiles

This should create ``plots/most_profiles.png`` without errors.


Phase 1: Config system, synthetic data, interface
-------------------------------------------------

Config parser — load from YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from bldfm.config_parser import load_config, BLDFMConfig

    config = load_config("examples/configs/multitower.yaml")

    print(f"Type: {type(config).__name__}")
    print(f"Domain: nx={config.domain.nx}, ny={config.domain.ny}, nz={config.domain.nz}")
    print(f"Domain ref: ({config.domain.ref_lat}, {config.domain.ref_lon})")
    print(f"Solver: closure={config.solver.closure}, footprint={config.solver.footprint}")
    print(f"Output: format={config.output.format}, dir={config.output.directory}")
    print(f"Parallel: workers={config.parallel.max_workers}, cache={config.parallel.use_cache}")

    assert isinstance(config, BLDFMConfig)
    assert config.domain.nx == 128
    assert config.solver.footprint is True
    assert config.met.n_timesteps == 3
    print("Phase 1 — load_config: OK")

Config parser — parse_config_dict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also build a config from a plain Python dictionary:

.. code-block:: python

    from bldfm.config_parser import parse_config_dict

    config2 = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [{"name": "test_tower", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0}],
        "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
        "solver": {"closure": "MOST", "footprint": True},
    })

    print(f"Tower local coords: x={config2.towers[0].x:.1f}, y={config2.towers[0].y:.1f}")
    assert config2.towers[0].x != 0.0 or config2.towers[0].y != 0.0
    print("Phase 1 — parse_config_dict + local coords: OK")

Tower lat/lon to local x/y
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``ref_lat`` and ``ref_lon`` are set, each tower's local ``(x, y)``
coordinates are computed automatically from its ``(lat, lon)``:

.. code-block:: python

    for t in config.towers:
        print(f"  {t.name}: lat={t.lat}, lon={t.lon} -> x={t.x:.1f} m, y={t.y:.1f} m")

MetConfig timeseries
^^^^^^^^^^^^^^^^^^^^

The ``MetConfig`` class supports both scalar and list-valued fields.  List
values represent a timeseries:

.. code-block:: python

    print(f"n_timesteps: {config.met.n_timesteps}")
    step0 = config.met.get_step(0)
    print(f"Step 0: {step0}")

    assert config.met.n_timesteps == 3
    print("Phase 1 — MetConfig: OK")

CLI — dry-run
^^^^^^^^^^^^^

.. code-block:: bash

    $ bldfm run examples/configs/multitower.yaml --dry-run

This should print the config summary (domain, towers with local x/y, number of
timesteps) and exit without running the solver.

Synthetic data — timeseries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from bldfm.synthetic import generate_synthetic_timeseries

    met = generate_synthetic_timeseries(n_timesteps=24, seed=42)

    print(f"Keys: {list(met.keys())}")
    print(f"Timestamps: {len(met['timestamps'])}")
    print(f"ustar range: [{min(met['ustar']):.3f}, {max(met['ustar']):.3f}]")
    print(f"MOL range: [{min(met['mol']):.1f}, {max(met['mol']):.1f}]")

    assert len(met["ustar"]) == 24
    assert len(met["timestamps"]) == 24
    print("Phase 1 — generate_synthetic_timeseries: OK")

Synthetic data — tower layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three spatial layouts are available: ``"grid"``, ``"transect"``, ``"random"``.

.. code-block:: python

    from bldfm.synthetic import generate_towers_grid

    for layout in ["grid", "transect", "random"]:
        towers = generate_towers_grid(n_towers=4, z_m=10.0, layout=layout, seed=42)
        print(f"\n{layout} layout:")
        for t in towers:
            print(f"  {t['name']}: ({t['lat']:.6f}, {t['lon']:.6f})")

    assert len(towers) == 4
    print("Phase 1 — generate_towers_grid: OK")

High-level interface — run_bldfm_single
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run_bldfm_single`` encapsulates the 3-step manual workflow (wind fields,
vertical profiles, solver) into one config-driven call:

.. code-block:: python

    from bldfm import run_bldfm_single
    from bldfm.config_parser import parse_config_dict

    config_small = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "modes": [128, 64], "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [{"name": "tower_A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0}],
        "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
        "solver": {"closure": "MOST", "footprint": True},
    })

    result = run_bldfm_single(config_small, config_small.towers[0])

    print(f"Result keys: {sorted(result.keys())}")
    print(f"flx shape: {result['flx'].shape}")
    print(f"conc shape: {result['conc'].shape}")
    print(f"Tower: {result['tower_name']}")

    expected_keys = {"conc", "flx", "grid", "params", "timestamp", "tower_name"}
    assert set(result.keys()) == expected_keys
    assert result["flx"].shape == (64, 128)
    print("Phase 1 — run_bldfm_single: OK")


Phase 2: Timeseries and multi-tower
------------------------------------

run_bldfm_timeseries
^^^^^^^^^^^^^^^^^^^^

Runs the solver for all timesteps in the met config for a single tower:

.. code-block:: python

    from bldfm import run_bldfm_timeseries
    from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid
    from bldfm.config_parser import parse_config_dict

    met_ts = generate_synthetic_timeseries(n_timesteps=3, seed=42)
    config_ts = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "modes": [128, 64], "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [{"name": "tower_A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0}],
        "met": met_ts,
        "solver": {"closure": "MOST", "footprint": True},
    })

    ts_results = run_bldfm_timeseries(config_ts, config_ts.towers[0])

    print(f"Number of results: {len(ts_results)}")
    for i, r in enumerate(ts_results):
        print(f"  t={r['timestamp']}: flx max={r['flx'].max():.6f}")

    assert len(ts_results) == 3
    print("Phase 2 — run_bldfm_timeseries: OK")

run_bldfm_multitower
^^^^^^^^^^^^^^^^^^^^

Runs the solver for all towers and all timesteps:

.. code-block:: python

    from bldfm import run_bldfm_multitower

    towers_mt = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
    config_mt = parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "modes": [128, 64],
            "ref_lat": towers_mt[0]["lat"], "ref_lon": towers_mt[0]["lon"],
        },
        "towers": towers_mt,
        "met": generate_synthetic_timeseries(n_timesteps=2, seed=42),
        "solver": {"closure": "MOST", "footprint": True},
    })

    mt_results = run_bldfm_multitower(config_mt)

    print(f"Tower names: {list(mt_results.keys())}")
    for name, res_list in mt_results.items():
        print(f"  {name}: {len(res_list)} timesteps")

    assert len(mt_results) == 2
    for name in mt_results:
        assert len(mt_results[name]) == 2
    print("Phase 2 — run_bldfm_multitower: OK")

Multitower example script
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python examples/multitower_example.py

This should create ``plots/multitower_footprints.png``.


Phase 3: Caching, NetCDF I/O, parallel execution
-------------------------------------------------

Green's function cache
^^^^^^^^^^^^^^^^^^^^^^

The ``GreensFunctionCache`` stores solver outputs as SHA-256 keyed ``.npz``
files to avoid redundant solves:

.. code-block:: python

    from bldfm import GreensFunctionCache, run_bldfm_single
    import numpy as np
    import shutil

    cache_dir = ".bldfm_cache_tutorial"
    cache = GreensFunctionCache(cache_dir=cache_dir)
    print(f"Cache directory: {cache.cache_dir}")

    # First run (cache miss)
    result1 = run_bldfm_single(config_small, config_small.towers[0], cache=cache)
    print("First run complete (cache miss)")

    # Second run (cache hit — should be faster)
    result2 = run_bldfm_single(config_small, config_small.towers[0], cache=cache)
    print("Second run complete (cache hit)")

    print(f"Results identical: {np.allclose(result1['flx'], result2['flx'])}")

    cache_files = list(cache.cache_dir.glob("*.npz"))
    print(f"Cache files: {len(cache_files)}")
    assert len(cache_files) >= 1

    # Clean up
    cache.clear()
    shutil.rmtree(cache_dir)
    print("Phase 3 — GreensFunctionCache: OK")

NetCDF I/O
^^^^^^^^^^

Save multi-tower results to CF-1.8 compliant NetCDF and load them back:

.. code-block:: python

    from bldfm import save_footprints_to_netcdf, load_footprints_from_netcdf

    save_footprints_to_netcdf(mt_results, config_mt, "output/tutorial_footprints.nc")
    print("Saved to output/tutorial_footprints.nc")

    ds = load_footprints_from_netcdf("output/tutorial_footprints.nc")

    print(f"\nDimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Global attrs: {dict(ds.attrs)}")
    print(f"\nFootprint shape: {ds['footprint'].shape}")
    print(f"Tower names: {list(ds['tower'].values)}")
    print(f"CF Convention: {ds.attrs['Conventions']}")

    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "footprint" in ds.data_vars
    assert ds.sizes["tower"] == 2
    assert ds.sizes["time"] == 2
    ds.close()
    print("Phase 3 — NetCDF I/O: OK")

Parallel execution
^^^^^^^^^^^^^^^^^^

Distribute work across CPU cores using ``ProcessPoolExecutor``.  Three
strategies are available: ``"towers"``, ``"time"``, and ``"both"``.

.. code-block:: python

    from bldfm import run_bldfm_parallel

    par_results = run_bldfm_parallel(config_mt, max_workers=2, parallel_over="towers")

    print(f"Parallel (towers) — towers: {list(par_results.keys())}")
    for name, res_list in par_results.items():
        print(f"  {name}: {len(res_list)} timesteps")

    assert set(par_results.keys()) == set(mt_results.keys())
    print("Phase 3 — run_bldfm_parallel: OK")

The same interface supports ``parallel_over="time"`` (distribute timesteps) and
``parallel_over="both"`` (flatten all tower × timestep pairs).  Each subprocess
disables numba parallelism (``NUM_THREADS=1``) to avoid CPU oversubscription.


Phase 4: Plotting
-----------------

All code below uses ``matplotlib.use("Agg")``.  Remove that line if you want
interactive display.

.. code-block:: python

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

plot_footprint_field with contours
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from bldfm import plot_footprint_field

    fig, ax = plt.subplots()
    ax = plot_footprint_field(
        result1["flx"], result1["grid"],
        ax=ax,
        contour_pcts=[0.5, 0.8],
        title="Footprint with 50% and 80% contours",
    )
    plt.savefig("plots/tutorial_contours.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/tutorial_contours.png")

You should see a pcolormesh with two dashed contour lines.

extract_percentile_contour
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the contour level and enclosed area for a given percentile:

.. code-block:: python

    from bldfm import extract_percentile_contour

    level_50, area_50 = extract_percentile_contour(result1["flx"], result1["grid"], pct=0.5)
    level_80, area_80 = extract_percentile_contour(result1["flx"], result1["grid"], pct=0.8)

    print(f"50% contour: level={level_50:.6f}, area={area_50:.0f} m^2")
    print(f"80% contour: level={level_80:.6f}, area={area_80:.0f} m^2")

    assert area_80 > area_50, "80% area should be larger than 50%"
    assert level_50 > level_80, "50% level should be higher (more restrictive)"
    print("Phase 4 — extract_percentile_contour: OK")

plot_footprint_on_map (optional: contextily)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overlays footprint contours and tower markers on map tiles:

.. code-block:: python

    try:
        from bldfm import plot_footprint_on_map

        fig, ax = plt.subplots(figsize=(10, 8))
        ax = plot_footprint_on_map(
            result1["flx"], result1["grid"], config_small,
            tower=config_small.towers[0],
            contour_pcts=[0.5, 0.8],
            title="Footprint on map",
        )
        plt.savefig("plots/tutorial_map.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plots/tutorial_map.png")
    except ImportError as e:
        print(f"Skipping map plot (install contextily): {e}")

plot_footprint_on_map with land cover (optional: owslib)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overlays footprint on ESA WorldCover 2021 land cover classes instead of
street map tiles.  Useful for interpreting what surface types (forest,
cropland, water, etc.) fall within the footprint source area:

.. code-block:: python

    try:
        from bldfm import plot_footprint_on_map

        fig, ax = plt.subplots(figsize=(10, 8))
        ax = plot_footprint_on_map(
            result1["flx"], result1["grid"], config_small,
            tower=config_small.towers[0],
            contour_pcts=[0.5, 0.8],
            land_cover=True,
            title="Footprint on land cover",
        )
        plt.savefig("plots/tutorial_landcover.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plots/tutorial_landcover.png")
    except ImportError as e:
        print(f"Skipping land cover plot (install owslib): {e}")

plot_wind_rose (optional: windrose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    try:
        from bldfm import plot_wind_rose

        fig = plt.figure()
        ax = plot_wind_rose(
            met_ts["wind_speed"], met_ts["wind_dir"],
            title="Synthetic wind rose",
        )
        plt.savefig("plots/tutorial_windrose.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plots/tutorial_windrose.png")
    except ImportError as e:
        print(f"Skipping wind rose (install windrose): {e}")

plot_footprint_timeseries
^^^^^^^^^^^^^^^^^^^^^^^^^

Track how the footprint extent evolves across timesteps:

.. code-block:: python

    from bldfm import plot_footprint_timeseries

    fig, ax = plt.subplots()
    ax = plot_footprint_timeseries(
        ts_results, ts_results[0]["grid"],
        pcts=[0.5, 0.8],
        title="Footprint area evolution",
    )
    plt.savefig("plots/tutorial_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/tutorial_timeseries.png")
    print("Phase 4 — plot_footprint_timeseries: OK")

plot_footprint_interactive (optional: plotly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    try:
        from bldfm import plot_footprint_interactive

        fig = plot_footprint_interactive(
            result1["flx"], result1["grid"],
            title="Interactive footprint",
        )
        fig.write_html("plots/tutorial_interactive.html")
        print("Saved plots/tutorial_interactive.html — open in browser")
    except ImportError as e:
        print(f"Skipping interactive plot (install plotly): {e}")


Phase 5: Documentation and tests
---------------------------------

Sphinx build
^^^^^^^^^^^^

.. code-block:: bash

    $ cd docs
    $ sphinx-build -W -b html source build/html

The ``-W`` flag treats warnings as errors.  The build should complete with zero
warnings.  To preview:

.. code-block:: bash

    $ python -m http.server 8000 -d build/html
    # Open http://localhost:8000

Full test suite
^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python -m pytest tests/ -v

All tests should pass.


Summary checklist
-----------------

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Phase
     - Feature
     - Verification
   * - 0
     - ``initialize()`` explicit
     - ``_initialized`` flips to ``True``; ``logs/``, ``plots/`` created
   * - 0
     - ``vertical_profiles`` 5-tuple
     - ``len(profiles) == 5``
   * - 0
     - ``plot_profiles.py`` compatibility
     - Script runs; ``most_profiles.png`` created
   * - 1
     - ``load_config()``
     - Returns ``BLDFMConfig``; fields match YAML
   * - 1
     - ``parse_config_dict()``
     - Dict → config; tower local coords computed
   * - 1
     - Tower lat/lon → local x/y
     - Non-zero ``x``, ``y`` values
   * - 1
     - ``MetConfig`` timeseries
     - ``n_timesteps``, ``get_step()`` return scalar dicts
   * - 1
     - CLI ``--dry-run``
     - Prints summary, no solver execution
   * - 1
     - ``generate_synthetic_timeseries()``
     - Correct length; ustar/MOL in expected ranges
   * - 1
     - ``generate_towers_grid()``
     - grid / transect / random produce distinct layouts
   * - 1
     - ``run_bldfm_single()``
     - Returns dict with 6 keys; ``flx.shape == (ny, nx)``
   * - 2
     - ``run_bldfm_timeseries()``
     - ``len(results) == n_timesteps``
   * - 2
     - ``run_bldfm_multitower()``
     - Dict of ``tower_name → [results]``
   * - 3
     - ``GreensFunctionCache``
     - ``.npz`` files created; cache hit matches miss
   * - 3
     - ``save/load_footprints_to_netcdf()``
     - CF-1.8; round-trip dimensions match
   * - 3
     - ``run_bldfm_parallel()``
     - Same structure as sequential; 3 strategies work
   * - 4
     - ``plot_footprint_field()``
     - Pcolormesh + dashed contour lines
   * - 4
     - ``extract_percentile_contour()``
     - ``area_80 > area_50``; ``level_50 > level_80``
   * - 4
     - ``plot_footprint_on_map()``
     - Map tiles + contours (if contextily installed)
   * - 4
     - ``plot_wind_rose()``
     - Polar wind rose (if windrose installed)
   * - 4
     - ``plot_footprint_timeseries()``
     - Line plot of area vs. timestep
   * - 4
     - ``plot_footprint_interactive()``
     - Plotly HTML (if plotly installed)
   * - 5
     - Sphinx build ``-W``
     - Zero warnings
   * - 5
     - ``pytest tests/ -v``
     - All tests pass
