Quickstart tutorial
===================

Get from installation to your first footprint plot in about 5 minutes.

Prerequisites
-------------

- Python >= 3.10 (a conda environment is recommended)
- The BLDFM repository cloned locally

All commands below assume you are in the repository root.


Step 1: Install
---------------

.. code-block:: bash

    $ pip install -e .

    # Optional: plotting extras (contextily, windrose, plotly)
    $ pip install -e ".[plotting]"

Verify the installation:

.. code-block:: bash

    $ python -c "import bldfm; print('BLDFM imported successfully')"


Step 2: Initialize the runtime
------------------------------

BLDFM requires an explicit ``initialize()`` call before running simulations.
This creates the ``logs/`` and ``plots/`` directories and configures logging.

.. code-block:: python

    import bldfm

    bldfm.initialize()

You should see ``logs/`` and ``plots/`` directories created in your working
directory, and a log file inside ``logs/``.


Step 3: Load a configuration
-----------------------------

BLDFM uses YAML configuration files to define the domain, towers, meteorology,
and solver settings.  A reference config is included in the repository:

.. code-block:: python

    from bldfm import load_config

    config = load_config("examples/configs/multitower.yaml")

    print(f"Domain: {config.domain.nx} x {config.domain.ny}, nz={config.domain.nz}")
    print(f"Towers: {[t.name for t in config.towers]}")
    print(f"Timesteps: {config.met.n_timesteps}")

Expected output::

    Domain: 128 x 384, nz=32
    Towers: ['tower_A', 'tower_B']
    Timesteps: 3


Step 4: Run the solver and plot
-------------------------------

Compute a footprint for the first tower at the first timestep, then visualise it:

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

You should see a 2D pcolormesh of an unstable footprint with visible lateral
dispersion and two dashed contour lines (50 % and 80 % cumulative footprint).
The footprint plume extends upwind of the tower position.


Step 5: Command-line alternative
---------------------------------

The same workflow is available from the shell via the ``bldfm`` CLI:

.. code-block:: bash

    # Validate the config without running the solver
    $ bldfm run examples/configs/multitower.yaml --dry-run

    # Run the full solve (all towers, all timesteps)
    $ bldfm run examples/configs/multitower.yaml

    # Run and save footprint plots to plots/
    $ bldfm run examples/configs/multitower.yaml --plot

The ``--dry-run`` flag prints the parsed configuration and exits.  The
``--plot`` flag saves a footprint PNG for each tower and timestep to ``plots/``.


Next steps
----------

- `Full Tutorial <tutorial_full.html>`_: Walk through **every** v1.0 feature
  step-by-step and verify it works.
- `Quick Reference <reference.html>`_: Concise code snippets for common
  workflows (multi-tower, parallel, NetCDF, plotting).
- `API Reference <src.html>`_: Detailed documentation for all modules and
  functions.
