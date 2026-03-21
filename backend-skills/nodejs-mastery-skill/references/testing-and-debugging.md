# Testing and Debugging Node.js Backends

## Test Layers

- domain logic tests
- route/transport tests
- adapter integration tests
- smoke tests for startup and shutdown
- failure tests for timeout and retry paths

## Debugging Questions

- is latency from event-loop lag or dependency slowness?
- is memory growth due to buffering, caching, or queue backlog?
- are duplicate effects caused by retries without idempotency?
- is one route performing too much orchestration?

## Review Checklist

- failure paths are tested
- timeouts are tested
- shutdown is tested
- logs identify dependency and request context
