# Tenant Fairness and Noisy Neighbor Playbook in Go Services

## Controls

- tenant-aware rate limits and quotas
- per-tenant concurrency or worker caps when justified
- tenant-level queue age and saturation visibility
- selective throttling of low-value tenant traffic

## Incident Questions

- which tenant is dominating pools, queues, or goroutines?
- where is tenant identity lost?
- what gets throttled first to protect shared health?

## Principal Heuristics

- Global rate limits are not enough when fairness matters.
- Tenant-aware signals should exist before on-call needs them.
