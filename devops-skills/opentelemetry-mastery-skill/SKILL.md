---
name: opentelemetry-principal-engineer
description: |
  Principal/Senior-level OpenTelemetry playbook for traces, metrics, logs, semantic conventions, collector pipelines, sampling, data governance, and production observability architecture.
  Use when: instrumenting services, designing collector topologies, standardizing telemetry, debugging context propagation, or building large-scale observability platforms.
---

# OpenTelemetry Mastery (Senior → Principal)

## Operate

- Start from what must be explainable in production: request flow, failure attribution, dependency latency, and causality.
- Treat telemetry schemas, propagation, sampling, and collector design as platform architecture.
- Prefer standardized semantic conventions over one-off instrumentation creativity.
- Design for correlation across traces, metrics, and logs without uncontrolled cost.

## Default Standards

- Context propagation must be reliable across service and async boundaries.
- Sampling policy should match debugging and business-critical workflows.
- Collector pipelines should be simple enough to operate and debug.
- Telemetry attributes are a cost and governance budget.
- Vendor portability should not mean lowest-common-denominator blindness.

## References

- Tracing model and span design: [references/tracing-model-and-span-design.md](references/tracing-model-and-span-design.md)
- Context propagation and async boundaries: [references/context-propagation-and-async-boundaries.md](references/context-propagation-and-async-boundaries.md)
- Semantic conventions and schema governance: [references/semantic-conventions-and-schema-governance.md](references/semantic-conventions-and-schema-governance.md)
- Sampling strategy and cost control: [references/sampling-strategy-and-cost-control.md](references/sampling-strategy-and-cost-control.md)
- Collector architecture and pipelines: [references/collector-architecture-and-pipelines.md](references/collector-architecture-and-pipelines.md)
- Metrics, logs, and correlation: [references/metrics-logs-and-correlation.md](references/metrics-logs-and-correlation.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
