# Reliability and Operations (OpenTelemetry)

## Operational Defaults

- Monitor collector queueing, memory, dropped telemetry, exporter latency, and pipeline error rates.
- Keep collector config reviewable and rollout-safe.
- Document vendor-specific assumptions even when using standards-based instrumentation.
- Distinguish instrumentation regressions from transport or backend failures.

## Run-the-System Thinking

- Telemetry pipelines deserve their own SLOs if the organization depends on them heavily.
- Upgrades across SDKs, collectors, and backends should be staged deliberately.
- On-call needs visibility into what was dropped, transformed, delayed, or sampled out.
- Multi-tenant telemetry platforms need policy before they need more pipelines.

## Principal Review Lens

- What failure mode hides the most truth while appearing healthy?
- Can teams detect silent telemetry loss quickly?
- Which upgrade path is most operationally dangerous right now?
- Are we operating a platform or accumulating observability plumbing debt?
