# CI/CD and Supply Chain (Docker)

## Rules

- Builds should be reproducible, signed where appropriate, and traceable to source.
- Registry workflows need retention, provenance, and policy.
- Secrets in CI are part of the threat model.
- Promotion between environments should be explicit.

## Principal Review Lens

- Can we prove which source built this image?
- Where can malicious or accidental image drift enter the pipeline?
- Is environment promotion safe and auditable?
