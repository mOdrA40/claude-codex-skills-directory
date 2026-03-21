# Runtime Operations (Docker)

## Rules

- Containers need sane signals, health behavior, logs, and restart posture.
- Startup and shutdown behavior should be intentional.
- Runtime defaults should match orchestrator expectations.
- Debug path should not rely on breaking immutability casually.

## Principal Review Lens

- How does this container fail and recover in production?
- Are signals and graceful shutdown handled correctly?
- Can operators diagnose issues without modifying the image?
