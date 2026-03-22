# Tenant Fairness and Noisy Neighbor Playbook in Zig Services

## Controls

- tenant-aware request and queue limits
- per-tenant caps on expensive parsing or worker usage where justified
- tenant-level visibility into memory, queue age, and failure rate
- selective throttling of low-value traffic first

## Incident Questions

- which tenant dominates memory or concurrency?
- where is tenant identity lost?
- what gets throttled first to preserve shared health?

## Principal Heuristics

- Fairness must include resource usage, not only authorization boundaries.
- If one tenant can spend shared memory budget, isolation is weak.
