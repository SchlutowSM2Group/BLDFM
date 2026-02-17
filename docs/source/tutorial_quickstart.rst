Quickstart tutorial
===================

BLDFM computes flux footprints and atmospheric dispersion fields for
eddy-covariance tower networks by solving the steady-state advection-diffusion
equation.  This guide gets you from installation to your first footprint plot
in about 5 minutes.

**Prerequisites:** Python >= 3.10 (a conda environment is recommended) and the
BLDFM repository cloned locally.  All commands below assume you are in the
repository root.


Step 1: Install
---------------

.. code-block:: bash

    $ pip install -e .

    # Optional: plotting extras (contextily, windrose, plotly)
    $ pip install -e ".[plotting]"

Verify the installation:

.. code-block:: bash

    $ python -c "import bldfm; print('BLDFM imported successfully')"


Step 2: Initialize and load a config
-------------------------------------

.. code-block:: python

    import bldfm

    bldfm.initialize()

    config = bldfm.load_config("examples/configs/footprint.yaml")

    print(f"Domain: {config.domain.nx} x {config.domain.ny}, nz={config.domain.nz}")
    print(f"Tower: {config.towers[0].name} at z={config.towers[0].z_m} m")
    print(f"Mode: {'footprint' if config.solver.footprint else 'concentration'}")

Expected output::

    Domain: 512 x 256, nz=32
    Tower: tower_1 at z=10.0 m
    Mode: footprint

``initialize()`` creates the ``logs/`` and ``plots/`` directories and
configures logging.  It only runs once -- subsequent calls are no-ops.


Step 3: Run the solver and plot
-------------------------------

Compute a footprint and visualise it with 50 % and 80 % cumulative contours:

.. code-block:: python

    from bldfm import run_bldfm_single, plot_footprint_field
    import matplotlib
    matplotlib.use("Agg")          # use "TkAgg" or remove for interactive display
    import matplotlib.pyplot as plt

    result = run_bldfm_single(config, config.towers[0])

    print(f"Result keys: {sorted(result.keys())}")
    print(f"Footprint shape: {result['flx'].shape}")

    ax = plot_footprint_field(
        result["flx"], result["grid"],
        contour_pcts=[0.5, 0.8],
        title="My first BLDFM footprint",
    )
    plt.savefig("plots/tutorial_first_footprint.png", dpi=150, bbox_inches="tight")
    print("Saved plots/tutorial_first_footprint.png")

You should see a 2D pcolormesh of a neutral footprint with two dashed contour
lines (50 % and 80 % cumulative footprint).  The footprint plume extends
upwind of the tower position.


Step 4: Command-line alternative
---------------------------------

The same workflow is available from the shell via the ``bldfm`` CLI:

.. code-block:: bash

    # Validate the config without running the solver
    $ bldfm run examples/configs/footprint.yaml --dry-run

    # Run the full solve
    $ bldfm run examples/configs/footprint.yaml

    # Run and save footprint plots to plots/
    $ bldfm run examples/configs/footprint.yaml --plot

The ``--dry-run`` flag prints the parsed configuration and exits.  The
``--plot`` flag saves a footprint PNG for each tower and timestep to ``plots/``.


Next steps
----------

- `Full Tutorial <tutorial_full.html>`_: Learn the complete API -- configuration,
  timeseries, multi-tower, parallel execution, I/O, and all plotting functions.
- `Quick Reference <reference.html>`_: Concise code snippets for common
  workflows.
- `API Reference <src.html>`_: Detailed documentation for all modules and
  functions.
