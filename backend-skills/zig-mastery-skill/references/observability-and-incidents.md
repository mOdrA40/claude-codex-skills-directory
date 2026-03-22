# Observability and Incident Response in Zig Services

## Principle

Observability should help explain allocator pressure, dependency slowness, queue growth, and release regressions quickly. Logging alone is not enough.

## Minimum Signals by Failure Class

For request-serving services, collect at least:

- request rate, latency, and failure rate
- dependency latency, failure rate, and saturation
- queue depth, oldest job age, and worker lag
- memory pressure indicators and release version
- current in-flight requests and concurrency hotspots

If you cannot distinguish dependency collapse from local resource pressure, on-call response will be slower than it should be.

## Signals

At minimum, expose:

- request latency and failure rate
- dependency latency and saturation
- queue depth and worker lag
- memory pressure indicators
- release version and node identity

## Logging Guidance

Prefer structured logs with stable fields such as:

- `service`
- `route` or operation name
- `request_id`
- `trace_id` if present
- `tenant_id` when relevant
- `dependency`
- `release_version`

Logs should help answer what failed, where it failed, and whether this is localized or systemic.

## Incident Triage Questions

- did the regression begin after a release?
- is memory pressure caused by traffic shape, queue growth, or one hot endpoint?
- are failures concentrated on one dependency or spread across the service?
- what signal indicates overload before hard failure or crash?
- can operators identify whether degradation is safe or user-impacting?

## Review Questions

- can operators distinguish dependency failure from local resource pressure?
- what metric shows overload before hard failure?
- is release version visible during incidents?
