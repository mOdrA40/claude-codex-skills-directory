# Observability and Incident Response in Zig Services

## Principle

Observability should help explain allocator pressure, dependency slowness, queue growth, and release regressions quickly. Logging alone is not enough.

## Signals

At minimum, expose:

- request latency and failure rate
- dependency latency and saturation
- queue depth and worker lag
- memory pressure indicators
- release version and node identity

## Review Questions

- can operators distinguish dependency failure from local resource pressure?
- what metric shows overload before hard failure?
- is release version visible during incidents?
