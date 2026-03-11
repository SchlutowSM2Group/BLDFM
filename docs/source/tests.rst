.. _tests:

The ``tests`` subpackage
========================

All tests can be run with:

.. code-block:: bash

   $ pytest -v tests/

Tests are organised into five categories using pytest marks:

.. code-block:: bash

   $ pytest -v -m unit tests/          # fast, no solver
   $ pytest -v -m integration tests/   # solver-dependent
   $ pytest -v -m parallel tests/      # multiprocessing
   $ pytest -v -m plotting tests/      # matplotlib / plotly
   $ pytest -v -m regression tests/    # compare against saved references

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

Regression tests (``regression``)
----------------------------------

Regression tests compare solver outputs against saved ``.npz`` reference files
in ``tests/references/``.  They catch silent numerical drift that property-based
tests (positivity, mass conservation) would miss.

Five scenarios are checked: ``single_footprint``, ``source_area``, ``plume_3d``,
``timeseries_step0``, and ``timeseries_step2``.  All use tolerances of
``atol=1e-6``, ``rtol=1e-5`` (MOST closure).

To regenerate baselines after intentional solver changes:

.. code-block:: bash

   $ pytest --update-references tests/test_regression.py \
         tests/test_integration.py tests/test_interface.py
   $ git add tests/references/

.. note::

   The integration and interface test files must be included so that the
   session-scoped solver fixtures are triggered.

tests.test\_regression module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tests.test_regression
   :members:
   :undoc-members:
   :show-inheritance:
