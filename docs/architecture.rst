Architecture
============

The template separates training, artifact generation, and serving while keeping
the local development flow intentionally small.

Workflow
--------

1. ``mlops_template.train`` loads the scikit-learn breast cancer demo dataset.
2. The training job builds a standard scaler plus logistic regression pipeline.
3. Evaluation metrics are written to ``artifacts/metrics.json``.
4. The model is serialized to ``artifacts/model.joblib``.
5. Version metadata is written to ``artifacts/model_version.json``.
6. A model card is generated at ``artifacts/MODEL_CARD.md``.
7. ``mlops_template.api`` loads the model and exposes prediction endpoints.

Model Metadata
--------------

The generated metadata contains:

* semantic model version;
* creation timestamp;
* SHA-256 hash of the serialized model;
* training dataset identifier;
* model type;
* ordered feature names;
* evaluation metrics.

Production Adaptation
---------------------

For a real project, replace the built-in dataset and local artifacts with
versioned data and managed storage:

* MLflow Tracking Server plus object storage;
* DVC with a remote backend;
* SageMaker Model Registry, Vertex AI Model Registry, or Azure ML Registry;
* a CI/CD promotion step that gates deployment on metrics and approval.
