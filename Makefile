.PHONY: install test train serve docs docs-clean docker-build clean

install:
	python -m pip install --upgrade pip
	pip install -e '.[dev,tracking]'

test:
	ruff check .
	mypy src
	pytest -q --cov=src --cov-report=term-missing

train:
	python -m mlops_template.train

serve:
	uvicorn mlops_template.api:app --host 0.0.0.0 --port 8000

docs:
	sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build

docker-build:
	docker build -t ai-mlops-template:local .

clean:
	rm -rf artifacts mlruns docs/_build .pytest_cache .ruff_cache .mypy_cache
