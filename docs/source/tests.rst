.. _tests:

The ``tests`` subpackage
========================

All tests can be run with:

.. code-block:: bash

   $ pytest -v tests/

Tests are organised into four categories using pytest marks.  Each category
runs as a separate CI job:

.. code-block:: bash

   $ pytest -v -m unit tests/          # fast, no solver
   $ pytest -v -m integration tests/   # solver-dependent
   $ pytest -v -m parallel tests/      # multiprocessing
   $ pytest -v -m plotting tests/      # matplotlib / plotly

Unit tests (``unit``)
---------------------

tests.test\_pbl\_model module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_pbl_model
   :members:
   :undoc-members:
   :show-inheritance:

tests.test\_config module
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_config
   :members:
   :undoc-members:
   :show-inheritance:

tests.test\_synthetic module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_synthetic
   :members:
   :undoc-members:
   :show-inheritance:

tests.test\_cache module
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_cache
   :members:
   :undoc-members:
   :show-inheritance:

Integration tests (``integration``)
------------------------------------

tests.test\_integration module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_integration
   :members:
   :undoc-members:
   :show-inheritance:

tests.test\_interface module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_interface
   :members:
   :undoc-members:
   :show-inheritance:

tests.test\_io module
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_io
   :members:
   :undoc-members:
   :show-inheritance:

Parallel tests (``parallel``)
-----------------------------

tests.test\_parallel module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_parallel
   :members:
   :undoc-members:
   :show-inheritance:

Plotting tests (``plotting``)
-----------------------------

tests.test\_plotting module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_plotting
   :members:
   :undoc-members:
   :show-inheritance:
