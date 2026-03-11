.. _src:

The ``bldfm`` package
=====================

This package contains the BLDFM model. It includes core modules for the PBL model, spectral solver, and utility functions, as well as higher-level modules for configuration, interface, caching, and synthetic data generation.  Plotting and I/O are provided by ``abltk``.

Core modules
------------

bldfm.pbl\_model module
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.pbl_model
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.solver module
^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.solver
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.utils module
^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.utils
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.fft\_manager module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.fft_manager
   :members:
   :undoc-members:
   :show-inheritance:

Configuration and interface
---------------------------

bldfm.config\_parser module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.config_parser
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.interface module
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.interface
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.cli module
^^^^^^^^^^^^^^^^

.. automodule:: bldfm.cli
   :members:
   :undoc-members:
   :show-inheritance:

Data and I/O
------------

bldfm.synthetic module
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.synthetic
   :members:
   :undoc-members:
   :show-inheritance:

bldfm.cache module
^^^^^^^^^^^^^^^^^^

.. automodule:: bldfm.cache
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

Plotting functions have moved to ``abltk.plotting``.  See the
`abl-tk documentation <https://github.com/SchlutowSM2Group/abl-tk>`_ for the
full API reference.  Common imports::

    from abltk.plotting import plot_footprint_field, plot_footprint_on_map
