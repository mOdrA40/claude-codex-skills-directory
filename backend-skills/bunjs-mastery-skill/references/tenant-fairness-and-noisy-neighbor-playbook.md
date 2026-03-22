# Tenant Fairness and Noisy Neighbor Playbook in Bun Services

## Controls

- tenant-aware rate limits
- per-tenant caps for expensive routes or workers
- tenant-aware backlog visibility
- selective degradation of low-value traffic

## Incident Questions

- which tenant dominates queue age or dependency calls?
- where is tenant identity lost in background processing?
- what gets throttled first to protect shared health?

## Principal Heuristics

- Fast runtimes still need fairness policy.
- If one tenant can spend everyone else's budget, isolation is weak.
