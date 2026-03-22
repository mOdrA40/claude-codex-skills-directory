# Tenant Fairness and Noisy Neighbor Playbook in Rust Services

## Controls

- tenant-aware quotas and rate limits
- per-tenant caps on expensive jobs or concurrency where justified
- tenant-level backlog, pool, and error visibility
- selective throttling of low-value traffic first

## Incident Questions

- which tenant dominates locks, pools, or queue age?
- where is tenant identity lost across async paths?
- what gets throttled first to protect shared health?

## Principal Heuristics

- Fairness is part of reliability, not only product policy.
- If one tenant can consume shared safety margin, isolation is weak.
