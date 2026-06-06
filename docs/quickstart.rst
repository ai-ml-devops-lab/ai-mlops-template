Quickstart
==========

Create an environment and install the project in editable mode:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,tracking,docs]"

On Windows:

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev,tracking,docs]"

Run the local checks:

.. code-block:: bash

   make test

Train the demo model:

.. code-block:: bash

   make train

Start the API:

.. code-block:: bash

   make serve

The OpenAPI documentation is available at http://localhost:8000/docs.

Build this documentation:

.. code-block:: bash

   make docs

The generated HTML output is written to ``docs/_build/html``.

Configuration
-------------

The workflow is configured with environment variables. See :doc:`reference`
for the complete field-level reference.

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Purpose
   * - ``ARTIFACT_DIR``
     - ``artifacts``
     - Directory for generated model cards and auxiliary artifacts.
   * - ``MODEL_PATH``
     - ``artifacts/model.joblib``
     - Serialized model artifact path.
   * - ``METRICS_PATH``
     - ``artifacts/metrics.json``
     - Metrics output path.
   * - ``VERSION_PATH``
     - ``artifacts/model_version.json``
     - Model metadata output path.
   * - ``MODEL_VERSION``
     - ``0.1.0``
     - Version label embedded in metadata and API responses.
   * - ``MLFLOW_TRACKING_URI``
     - ``file:./mlruns``
     - MLflow tracking destination when MLflow is installed.
