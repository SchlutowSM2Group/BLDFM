.. _runs:

Example scripts and workflows
=============================

BLDFM provides three tiers of example scripts:

- **``examples/``** — Config-driven, high-level interface (start here)
- **``runs/low_level/``** — Direct API calls for power users
- **``runs/manuscript/``** — Paper reproduction scripts using config + plotting library


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


Low-level examples (``runs/low_level/``)
-----------------------------------------

These scripts call ``vertical_profiles``, ``ideal_source``, and
``steady_state_transport_solver`` directly, giving full control over every parameter.

runs.low\_level.minimal\_example module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.low_level.minimal_example
   :members:
   :undoc-members:
   :show-inheritance:

runs.low\_level.footprint\_example module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.low_level.footprint_example
   :members:
   :undoc-members:
   :show-inheritance:

runs.low\_level.plot\_profiles module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.low_level.plot_profiles
   :members:
   :undoc-members:
   :show-inheritance:


Manuscript figures (``runs/manuscript/``)
------------------------------------------

These scripts reproduce the figures in the BLDFM paper.  They use the config-driven
interface with ``dataclasses.replace()`` for parameter mutation and the plotting library
for figure generation.

To regenerate all manuscript figures:

.. code-block:: bash

    $ python -m runs.manuscript.generate_all

runs.manuscript.comparison\_analytic module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.comparison_analytic
   :members:
   :undoc-members:
   :show-inheritance:

runs.manuscript.comparison\_footprint\_unstable module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.comparison_footprint_unstable
   :members:
   :undoc-members:
   :show-inheritance:

runs.manuscript.comparison\_footprint\_neutral module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.comparison_footprint_neutral
   :members:
   :undoc-members:
   :show-inheritance:

runs.manuscript.comparison\_footprint\_stable module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.comparison_footprint_stable
   :members:
   :undoc-members:
   :show-inheritance:

runs.manuscript.analytic\_convergence\_test module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.analytic_convergence_test
   :members:
   :undoc-members:
   :show-inheritance:

runs.manuscript.numeric\_convergence\_test module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: runs.manuscript.numeric_convergence_test
   :members:
   :undoc-members:
   :show-inheritance:
