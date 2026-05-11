FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app
RUN useradd --create-home appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip     && pip install --no-cache-dir .     && python -m mlops_template.train

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"
CMD ["uvicorn", "mlops_template.api:app", "--host", "0.0.0.0", "--port", "8000"]
