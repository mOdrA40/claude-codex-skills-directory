# Incident Runbooks (Redis)

## Rules

- Cover eviction storms, memory exhaustion, failover issues, hot keys, and cache stampede.
- Stabilize user-facing correctness before optimizing latency.
- Include safe emergency actions and forbidden commands.
- Tie recovery to observable metrics and app behavior.

## Principal Review Lens

- Can on-call stop cascading misses quickly?
- Which action risks data loss or wider outage?
- What confirms true recovery instead of temporary calm?
