# Survivability and Failures (CockroachDB)

## Rules

- Regional/node failure should be an exercised scenario, not a hopeful one.
- Distinguish degraded latency from correctness failure.
- Recovery plans must include application behavior under retry pressure.
- Measure recovery time and blast radius per topology.

## Principal Review Lens

- What breaks first under region impairment?
- Which workloads degrade gracefully and which fail hard?
- Are operators trained on likely failure sequences?
