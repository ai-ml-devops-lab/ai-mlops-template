Operations
==========

Release Readiness
-----------------

Before promoting a model or service image:

* run the test suite and static checks;
* confirm metrics exceed the agreed threshold;
* verify the model card exists and documents limitations;
* build the Docker image;
* scan the image in the target environment;
* record the previous deployable model artifact for rollback.

Useful Commands
---------------

.. code-block:: bash

   make train
   make test
   make docs
   docker compose up --build

API Smoke Test
--------------

After the API starts, check health:

.. code-block:: bash

   curl http://localhost:8000/health

Inspect the feature contract:

.. code-block:: bash

   curl http://localhost:8000/features

Read model metadata:

.. code-block:: bash

   curl http://localhost:8000/metadata

Rollback Notes
--------------

The API loads the model from ``MODEL_PATH``. A rollback can be performed by
pointing ``MODEL_PATH`` and ``VERSION_PATH`` back to a previously approved
artifact bundle, then restarting the service.
