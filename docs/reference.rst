Reference
=========

This page documents the variable names, artifact fields, and API payload fields
used by the template.

Environment Variables
---------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Default
     - Used by
     - Description
   * - ``ARTIFACT_DIR``
     - ``artifacts``
     - training
     - Base directory for generated auxiliary artifacts, including the model card.
   * - ``MODEL_PATH``
     - ``artifacts/model.joblib``
     - training, API
     - Path where the trained scikit-learn pipeline is written and loaded from.
   * - ``METRICS_PATH``
     - ``artifacts/metrics.json``
     - training
     - Path where evaluation metrics are written as JSON.
   * - ``VERSION_PATH``
     - ``artifacts/model_version.json``
     - training, API
     - Path where versioned model metadata is written and read from.
   * - ``MODEL_VERSION``
     - ``0.1.0``
     - training, API, MLflow
     - Version label embedded in model metadata, API responses, and MLflow runs.
   * - ``MLFLOW_TRACKING_URI``
     - ``file:./mlruns``
     - training
     - MLflow tracking destination when the optional MLflow dependency is installed.

Generated Artifacts
-------------------

.. list-table::
   :header-rows: 1

   * - Path
     - Format
     - Description
   * - ``artifacts/model.joblib``
     - Joblib
     - Serialized scikit-learn pipeline containing preprocessing and classifier steps.
   * - ``artifacts/metrics.json``
     - JSON
     - Evaluation metrics from the test split.
   * - ``artifacts/model_version.json``
     - JSON
     - Versioned metadata for traceability and API metadata responses.
   * - ``artifacts/MODEL_CARD.md``
     - Markdown
     - Human-readable model summary with intended use, metrics, and limitations.

Model Metadata Fields
---------------------

``artifacts/model_version.json`` and ``GET /metadata`` expose these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``model_version``
     - string
     - Version label from ``MODEL_VERSION``.
   * - ``created_at``
     - string
     - UTC ISO-8601 timestamp for the training run.
   * - ``model_sha256``
     - string
     - SHA-256 hash of the serialized model artifact.
   * - ``training_dataset``
     - string
     - Dataset identifier used during training.
   * - ``model_type``
     - string
     - Human-readable model family, currently ``logistic_regression``.
   * - ``feature_names``
     - list[string]
     - Ordered feature contract expected by the prediction API.
   * - ``metrics``
     - object
     - Evaluation metrics and split sizes.

Metric Fields
-------------

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``accuracy``
     - float
     - Share of correct predictions on the test split.
   * - ``f1``
     - float
     - Binary F1 score on the test split.
   * - ``roc_auc``
     - float
     - ROC AUC based on positive-class probabilities.
   * - ``n_train``
     - integer
     - Number of training rows.
   * - ``n_test``
     - integer
     - Number of test rows.
   * - ``n_features``
     - integer
     - Number of input features.

Prediction Payloads
-------------------

``PredictionRequest``
   Request body for ``POST /predict``.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Constraints
     - Description
   * - ``features``
     - list[float]
     - exactly 30 finite numbers
     - Input feature vector in the order documented below.

``PredictionResponse``
   Response body from ``POST /predict``.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``prediction``
     - integer
     - Predicted class label, either ``0`` or ``1``.
   * - ``probability_positive``
     - float
     - Positive-class probability in the inclusive range ``0.0`` to ``1.0``.
   * - ``model_version``
     - string
     - Version label of the model used for the prediction.

Feature Order
-------------

The ``features`` list must use this exact order:

.. list-table::
   :header-rows: 1

   * - Index
     - Feature name
   * - 0
     - ``mean radius``
   * - 1
     - ``mean texture``
   * - 2
     - ``mean perimeter``
   * - 3
     - ``mean area``
   * - 4
     - ``mean smoothness``
   * - 5
     - ``mean compactness``
   * - 6
     - ``mean concavity``
   * - 7
     - ``mean concave points``
   * - 8
     - ``mean symmetry``
   * - 9
     - ``mean fractal dimension``
   * - 10
     - ``radius error``
   * - 11
     - ``texture error``
   * - 12
     - ``perimeter error``
   * - 13
     - ``area error``
   * - 14
     - ``smoothness error``
   * - 15
     - ``compactness error``
   * - 16
     - ``concavity error``
   * - 17
     - ``concave points error``
   * - 18
     - ``symmetry error``
   * - 19
     - ``fractal dimension error``
   * - 20
     - ``worst radius``
   * - 21
     - ``worst texture``
   * - 22
     - ``worst perimeter``
   * - 23
     - ``worst area``
   * - 24
     - ``worst smoothness``
   * - 25
     - ``worst compactness``
   * - 26
     - ``worst concavity``
   * - 27
     - ``worst concave points``
   * - 28
     - ``worst symmetry``
   * - 29
     - ``worst fractal dimension``
