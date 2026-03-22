# Tenant Fairness and Noisy Neighbor Playbook in Node.js Services

## Goal

Protect shared latency, concurrency, and dependency budgets from one tenant or workload shape dominating the system.

## Controls

- tenant-aware rate limits
- per-tenant queue or concurrency ceilings when justified
- traffic classification for expensive routes
- visibility into tenant-level error rate, lag, and saturation

## Incident Questions

- is one tenant dominating event loop time or queue age?
- can low-value tenant traffic be throttled first?
- where is tenant identity lost across async boundaries?

## Principal Heuristics

- Global protection without fairness policy still permits noisy-neighbor harm.
- Tenant fairness should be observable before it becomes an incident.
