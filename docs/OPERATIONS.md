# Operations Runbook

## Release readiness checklist

- Tests pass.
- Model metrics exceed the agreed threshold.
- Model card exists and has limitations.
- Docker image builds.
- Security scan passes in the target environment.
- Rollback model artifact is known.

## Useful commands

```bash
make train
make test
docker compose up --build
```
