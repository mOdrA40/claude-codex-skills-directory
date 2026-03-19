# OpenTelemetry SDK Bootstrap (Service Template)

This doc complements `observability.md` (which covers `otelhttp` instrumentation).

## Goals

- traces + metrics export via OTLP
- propagators configured once
- graceful shutdown flushes telemetry

## Minimal bootstrap (shape)

Steps:
1) Build a `resource` with service name/version and environment.
2) Create OTLP exporters (trace/metric).
3) Set `TracerProvider` and `MeterProvider`.
4) Set text map propagator (W3C tracecontext + baggage; optionally autoprop via env).
5) Hook shutdown into graceful shutdown.

## Good vs bad

Bad: forget to shut down providers → spans dropped on exit.

Good: keep `shutdown()` and call it during service shutdown.

## Tip: start simple

If you’re new to OTel:
- traces first (highest debugging value)
- then RED metrics (requests, errors, duration)
- only then logs bridges (often noisy)

