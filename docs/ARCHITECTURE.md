# Architecture

```mermaid
flowchart LR
  D[Training data] --> T[Training job]
  T --> M[Model artifact]
  T --> V[Version metadata]
  T --> C[Model card]
  M --> A[FastAPI service]
  V --> A
  A --> U[Consumers]
```

## Production notes

- Replace the built-in dataset with a versioned data source.
- Run training in a controlled environment.
- Promote models only after automated metric gates and human approval.
- Keep the serving image immutable and trace it back to source commit and model hash.
