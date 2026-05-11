.PHONY: install test train serve docker-build clean

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

docker-build:
	docker build -t ai-mlops-template:local .

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache .mypy_cache
