Python API
==========

Training
--------

.. automodule:: mlops_template.train
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: mlops_template.config
   :members:
   :undoc-members:
   :show-inheritance:

Schemas
-------

.. automodule:: mlops_template.schema
   :members:
   :undoc-members:
   :show-inheritance:

HTTP API
--------

``GET /health``
   Returns service health, the configured model path, and whether the model
   artifact exists.

``GET /features``
   Returns the ordered feature contract required by ``POST /predict``.

``GET /metadata``
   Returns versioned model metadata. If metadata does not exist yet, the API
   triggers a local training run.

``POST /predict``
   Accepts a ``PredictionRequest`` with 30 numeric features and returns a
   ``PredictionResponse`` containing the predicted class, positive-class
   probability, and model version.
