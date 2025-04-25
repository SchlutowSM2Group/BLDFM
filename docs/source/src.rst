.. _src:

The ``src`` subpackage
======================

This subpackage contains the main code for the BLDFM model. It includes modules for the PBL model, the solver, and utility functions.

It is organized into three main modules:
   1. :py:mod:`src.pbl_model`: Computes vertical profiles of horizontal wind and eddy diffusivity in the planetary boundary layer using Monin-Obukhov Similarity Theory (MOST).
   2. :py:mod:`src.solver`: Solves the steady-state advection-diffusion equation for scalar concentration fields using FFT-based methods and numerical integration schemes.
   3. :py:mod:`src.utils`: Provides utility functions for preprocessing, synthetic input generation, and diagnostics.

Modules
-------

src.pbl\_model module
---------------------

.. automodule:: src.pbl_model
   :members:
   :undoc-members:
   :show-inheritance:

src.solver module
-----------------

.. automodule:: src.solver
   :members:
   :undoc-members:
   :show-inheritance:

src.utils module
----------------

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. Module contents
.. ---------------

.. .. automodule:: src
..    :members:
..    :undoc-members:
..    :show-inheritance:
