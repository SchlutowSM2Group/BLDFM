Quickstart guide
================

This guide provides a brief introduction to using the BLDFM framework through example scripts. It covers running a minimal example and extending it to compare with the Kormann-Meixner (FKM) footprint model.

Minimal Example
---------------

The :py:mod:`runs.minimal_example` script demonstrates the basic usage of the BLDFM framework. It calculates concentration and flux fields for a neutrally stratified boundary layer with default settings.

To run the script, execute the following command in the terminal:

.. code-block:: bash

    $ python3 -m runs.minimal_example

This will generate three plots saved in the `plots/` directory:

    * Concentration at z0: The concentration field at the surface.
    * Concentration at zm: The concentration field at the measurement height.
    * Vertical kinematic flux at zm: The flux field at the measurement height.

The script uses the following steps:
    1. Defines the domain, grid resolution, and atmospheric parameters (e.g., wind speed, roughness length).
    2. Generates a synthetic surface flux field using :py:func:`src.utils.ideal_source`.
    3. Computes vertical profiles of wind and diffusivity using :py:func:`src.pbl_model.vertical_profiles`.
    4. Solves the steady-state transport equation using :py:func:`src.solver.steady_state_transport_solver`.

This example provides a foundation for understanding the core components of BLDFM.

Comparison with the FKM Model
-----------------------------

The :py:mod:`runs.comparison_footprint` script extends the minimal example by comparing the BLDFM model with the Kormann-Meixner (FKM) footprint model. This script demonstrates how to evaluate the performance of BLDFM against an established analytical model.

To run the script, execute the following command in the terminal:

.. code-block:: bash

    $ python3 -m runs.comparison_footprint

This will generate a plot saved in the `plots/` directory:
    * Comparison of BLDFM and FKM footprints: A side-by-side visualization of the flux footprints from both models.

The script follows these steps:
    1. Sets up the domain, grid resolution, and atmospheric parameters, similar to the minimal example.
    2. Computes the BLDFM footprint using :py:func:`src.solver.steady_state_transport_solver`.
    3. Computes the FKM footprint using `estimateFootprint` from the Kormann Meixner model.
    4. Plots the results for visual comparison.

This example demonstrates how BLDFM can be used for model validation and comparison with analytical solutions.
