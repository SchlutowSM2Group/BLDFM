.. _runs:

Example scripts and workflows
=============================

BLDFM provides two tiers of example scripts:

- **``runs/low_level/``** — Direct API calls for power users
- **``runs/manuscript/``** — Paper reproduction scripts (both interface and low-level)


Low-level examples (``runs/low_level/``)
-----------------------------------------

These scripts call ``vertical_profiles``, ``ideal_source``, and
``steady_state_transport_solver`` directly, giving full control over every parameter.

.. code-block:: bash

    $ python runs/low_level/minimal_example.py
    $ python runs/low_level/footprint_example.py
    $ python runs/low_level/plot_profiles.py
    $ python runs/low_level/point_measurement_example.py
    $ python runs/low_level/source_area_example.py


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
