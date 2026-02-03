# Observability (Traces + Metrics + Logs)

## Golden Rule

Make a single request debuggable in 2 minutes:
- one trace ID (propagated across services)
- structured logs correlated to the trace
- RED metrics (Rate/Errors/Duration) per endpoint and dependency

## OpenTelemetry (Go) — net/http instrumentation (up-to-date)

For standard library `net/http`, OpenTelemetry Go Contrib provides `otelhttp` wrappers:
- server: wrap handlers via `otelhttp.NewHandler`
- client: wrap transports via `otelhttp.NewTransport`

Example (server wrapper):

```go
handler := otelhttp.NewHandler(mux, "my-http-service")
_ = http.ListenAndServe(":8080", handler)
```

Example (client wrapper):

```go
client := http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)}
```

Context propagation:
- Prefer W3C TraceContext + Baggage.
- If you must interop, you can configure propagators via env (e.g. `OTEL_PROPAGATORS`) using `autoprop`.

## Logging

- Use JSON structured logs in services.
- Define a stable field schema (`service`, `env`, `trace_id`, `span_id`, `request_id`, `user_id` (careful), `latency_ms`, `status`).
- Redact by design: create helpers so callers can’t accidentally log secrets.

## Metrics

- Prefer “few, high-signal” metrics:
  - HTTP: requests total, errors total, duration histogram
  - Dependencies: DB duration + error count, queue publish/consume duration, outbound HTTP duration
  - Saturation: goroutines, heap, GC pause, pool wait time, queue depth

