# Executors, Workers, and Platform Scaling

## Rules

- Executor choice affects cost, isolation, latency, and operational burden.
- Worker capacity and queueing strategy should align with workload classes.
- Scaling Airflow means scaling scheduler clarity as much as raw workers.
- Platform design should protect critical workflows from noisy neighbors.

## Practical Guidance

- Separate high-priority and exploratory workloads where needed.
- Track queueing, worker saturation, scheduler lag, and DB dependency health.
- Keep executor choice matched to team maturity and environment shape.
- Benchmark around real DAG behavior, not synthetic task counts.

## Principal Review Lens

- Which workload class is stressing the platform most?
- Are we scaling workers while ignoring scheduler or metadata bottlenecks?
- What executor assumption is least justified today?
- Which separation would most improve platform predictability?
