.. _runs:

Example scripts and workflows
=============================

BLDFM provides three tiers of example scripts:

- **``examples/``** — Config-driven, high-level interface (start here)
- **``examples/low_level/``** — Direct API calls for power users
- **``runs/manuscript/``** — Paper reproduction scripts (both interface and low-level)


High-level examples (``examples/``)
-------------------------------------

These scripts use YAML configs and the ``run_bldfm_single`` / ``run_bldfm_multitower``
interface.  Each example has a corresponding YAML config in ``examples/configs/``.

.. code-block:: bash

    $ python examples/minimal_example.py
    $ python examples/footprint_example.py
    $ python examples/parallel_example.py
    $ python examples/multitower_example.py
    $ python examples/3d_plume.py
    $ python examples/minimal_example_3d.py

    # Or use the CLI
    $ bldfm run examples/configs/multitower.yaml --plot


Low-level examples (``examples/low_level/``)
----------------------------------------------

These scripts call ``vertical_profiles``, ``ideal_source``, and
``steady_state_transport_solver`` directly, giving full control over every parameter.

.. code-block:: bash

    $ python examples/low_level/minimal_example.py
    $ python examples/low_level/footprint_example.py
    $ python examples/low_level/plot_profiles.py
    $ python examples/low_level/point_measurement_example.py


Manuscript figures (``runs/manuscript/``)
------------------------------------------

These scripts reproduce the figures in the BLDFM paper. They are provided in
two forms:

- **``runs/manuscript/interface/``** — Config-driven interface with ``dataclasses.replace()``
  for parameter mutation and the plotting library for figure generation.
- **``runs/manuscript/low_level/``** — Direct API calls matching the original manuscript code.

To regenerate all manuscript figures (both tiers):

.. code-block:: bash

    $ python runs/manuscript/generate_all.py

    # Or only one tier:
    $ python runs/manuscript/generate_all.py --tier interface
    $ python runs/manuscript/generate_all.py --tier low_level
