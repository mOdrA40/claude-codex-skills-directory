# Observability (Rust Services)

Observability exists to make one bad request or one degraded dependency diagnosable in minutes, not hours.

## Minimum Required Signals

- Structured logs with stable keys.
- RED metrics for inbound traffic.
- Dependency latency/error metrics.
- Distributed traces across HTTP, DB, queue, and background jobs.

## Logging Defaults

Include fields such as:

- `service`
- `env`
- `version`
- `trace_id`
- `request_id`
- `tenant_id` when relevant
- `status`
- `latency_ms`

Do not log:

- raw secrets,
- full auth headers,
- sensitive payloads by default,
- stack traces in client-facing outputs.

## Metrics Defaults

At minimum:

- request count
- error count
- duration histogram
- DB duration and pool wait time
- outbound HTTP duration and error count
- queue lag or worker backlog
- process saturation signals when relevant

## Tracing Defaults

- Propagate trace context across service boundaries.
- Instrument DB, outbound HTTP, and message-publish/consume boundaries.
- Use spans for expensive operations, not every tiny helper.
- Add domain identifiers as span fields when safe.

## Startup and Shutdown Signals

On startup, log:

- service name
- version / git SHA
- bind address
- environment
- critical config mode flags

On shutdown, log:

- reason for shutdown
- in-flight drain start/end
- timeout or forced-exit conditions

## Principal Review Lens

Ask:

- Can on-call isolate one failing request from logs and traces?
- Which metric turns red before customers complain?
- Are logs safe by default?
- Can we distinguish dependency failure from code regression quickly?
