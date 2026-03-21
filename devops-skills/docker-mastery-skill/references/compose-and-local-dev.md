# Compose and Local Dev

## Rules

- Local stacks should model production shape enough to catch real issues.
- Keep compose files understandable and environment-specific overrides explicit.
- Health checks and dependency assumptions matter in local dev too.
- Avoid turning Compose into an accidental production platform.

## Principal Review Lens

- Does local dev reproduce the failure modes that matter?
- Is the compose setup helping clarity or hiding complexity?
- Which local convenience creates false confidence?
