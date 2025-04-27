.. BLDFM documentation master file, created by
   sphinx-quickstart on Thu Apr 24 15:58:54 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BLDFM's Documentation!
==================================

The **Boundary Layer Dispersion and Footprint Model (BLDFM)** is a atmospheric dispersion and footprint model designed for microscale applications in the planetary boundary layer (PBL). It numerically solves the three-dimensional steady-state advection-diffusion equation in Eulerian form, providing robust tools for modeling scalar transport under various turbulent regimes.

This documentation serves as a guide to understanding, configuring, and extending BLDFM for your specific use cases. This documentation aims to be accessible to researchers, developers, or practitioners. If you spot a feature gap in BLDFM or space for improvement in the documentation, feel free to reach out with a contribution or an issue!


Key Features
------------

- **Numerical Solver**: Efficiently solves the steady-state advection-diffusion equation using Fourier transforms and numerical methods like the Semi-Implicit Euler (SIE) and Exponential Integrator (EI).
- **Atmospheric Stability**: Computes vertical profiles of mean wind and eddy diffusivity using Monin-Obukhov Similarity Theory (MOST).
- **Customizability**: Easily extendable with additional turbulence closures and parameterizations.
- **Validation**: Tested against analytical solutions and benchmark models, ensuring accuracy and reliability.
- **Example Workflows**: Includes pre-configured scripts for footprint modeling and dispersion analysis.


Getting Started
---------------

To begin using BLDFM, follow these steps in the root directory of the repository:

1. **Installation**:
   Install the package using pip:

   .. code-block:: bash

      $ pip install -e .

2. **Run Example Scripts**: 
   Explore the example workflows in the runs subpackage:

   .. code-block:: bash

      $ python -m runs.comparison_footprint

3. **Dive into the Documentation**: 
   Use the navigation links below to explore the package structure, API reference, and example workflows.


Navigation
----------

- `Quickstart Guide <quickstart.html>`_: A step-by-step guide to installing and running BLDFM.
- `Package Overview <overview.html>`_: Learn about the core modules and their interactions.
- `Example Workflows <runs.html>`_: Explore pre-configured scripts in the ``runs`` subpackage.
- `Test Suite <tests.html>`_: Understand how BLDFM ensures reliability through rigorous testing.
- `API Reference <src.html>`_: Detailed documentation for all core modules and functions.


About BLDFM
-----------

BLDFM is developed by Mark Schlutow, Ray Chew, and Mathias Göckede. It is licensed under the GNU License and is provided as an open-source software package to support research and development in atmospheric sciences.

For more details, visit the `GitHub repository <https://github.com/SchlutowSM2Group/BLDFM>`_.


.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:

   Home <self>
   quickstart
   overview
   API reference <src>
   GitHub repository <https://github.com/SchlutowSM2Group/BLDFM>


.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:

   runs
   tests


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   acknowledgments
   Glossary Index <genindex>
   Module Index <modindex>
