# Sidecars, Ambient Mode, and Dataplane Tradeoffs

## Rules

- Dataplane choice affects cost, latency, operability, and feature posture.
- Sidecars provide strong locality but add workload-level tax.
- Ambient-style designs change operational boundaries and debugging patterns.
- Teams must understand the consequences of dataplane choice before scaling adoption.

## Tradeoff Guidance

- Evaluate resource overhead, startup behavior, lifecycle complexity, and policy coverage.
- Consider which workloads are most sensitive to injected proxy overhead.
- Align dataplane choice with organizational ability to debug network behavior.
- Avoid mixing models without clear ownership and support guidance.

## Principal Review Lens

- What is the true operational tax of the chosen dataplane?
- Which class of failure becomes harder to debug?
- Are we choosing dataplane mode for platform goals or trend-chasing?
- What workload class should be exempt or handled differently?
