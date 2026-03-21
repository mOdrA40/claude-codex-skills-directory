# Dockerfiles and Build Patterns

## Rules

- Dockerfiles should be deterministic and boring.
- Layer ordering should optimize rebuild speed and cache hit rate.
- Avoid leaking secrets into layers or build args casually.
- Entrypoints and workdirs should be explicit.

## Principal Review Lens

- Which layer changes most often and why?
- Is this Dockerfile easy to audit under incident pressure?
- Could a build secret leak into the final image?
