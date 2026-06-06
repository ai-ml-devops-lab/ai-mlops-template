# AI MLOps Template

A production-style template for training, testing, tracking, versioning, and serving a machine learning model.

## What this solves for companies

Many organizations have useful notebooks but no repeatable path to production. This repository demonstrates:

- reproducible training code instead of notebook-only experiments;
- metric logging and artifact metadata;
- a versioned model bundle;
- automated tests and linting;
- generated English Sphinx documentation;
- an API for serving predictions;
- Docker packaging;
- release workflow for GitHub Container Registry.

## Local quickstart

```bash
python -m venv .venv
# linux (bash):
source .venv/bin/activate
# windows (cmd):
.venv\Scripts\activate
# linux (bash):
pip install -e '.[dev,tracking]'
# windows (cmd):
python -m pip install -e .[dev,tracking]
make test
make train
make serve
```

Then open `http://localhost:8000/docs`.

Build the project documentation with:

```bash
python -m pip install -e ".[docs]"
make docs
```

Then open `docs/_build/html/index.html`.

The Sphinx reference documents environment variables, artifact fields, API payload fields, metrics, and the required prediction feature order.

## Docker

```bash
docker build -t ai-mlops-template:local .
docker run --rm -p 8000:8000 ai-mlops-template:local
```

## Model lifecycle

1. `src/mlops_template/train.py` trains a scikit-learn model on a built-in dataset.
2. Metrics are written to `artifacts/metrics.json`.
3. The model is written to `artifacts/model.joblib`.
4. `artifacts/model_version.json` contains a content hash and semantic version.
5. `artifacts/MODEL_CARD.md` summarizes intended use, metrics, and limitations.
6. The API loads the model from `MODEL_PATH` or the default artifact location.

## API endpoints

- `GET /health` returns service health and artifact availability.
- `GET /features` returns the ordered feature contract.
- `GET /metadata` returns versioned model metadata.
- `POST /predict` returns the model prediction and positive-class probability.

## Production extensions

For a client project, replace local artifacts with one of:

- MLflow Tracking Server + S3-compatible object storage;
- DVC remote;
- cloud-native registry such as SageMaker Model Registry, Vertex AI Model Registry, or Azure ML Registry.

Keep the CI pipeline small for GitHub Free. Run long training on self-hosted or cloud runners when needed.
