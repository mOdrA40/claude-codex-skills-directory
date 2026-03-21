# Go Service Skeletons and Boundary Templates

## Purpose

This guide helps engineers choose the smallest Go backend structure that remains production-safe.

## Small Service Skeleton

Use when:

- one API surface
- one data store
- small team
- limited cross-service orchestration

```text
cmd/api/
internal/app/
internal/http/
internal/service/
internal/repository/
internal/domain/
```

## Medium Service Skeleton

Use when:

- multiple dependencies
- background workers exist
- the service has non-trivial policy and failure behavior

```text
cmd/api/
cmd/worker/
internal/bootstrap/
internal/transport/http/
internal/usecase/
internal/domain/
internal/ports/
internal/adapters/
internal/observability/
```

## Rules

- keep `main` thin
- own retries in one place
- map domain errors once
- do not let repositories become service layers
- keep goroutine ownership explicit

## Review Questions

- Is this service under-structured or over-structured for its size?
- Are transport concerns leaking inward?
- Can the worker path and API path fail independently?
