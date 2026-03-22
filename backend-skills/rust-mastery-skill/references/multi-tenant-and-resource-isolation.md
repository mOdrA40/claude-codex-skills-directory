# Multi-Tenant and Resource Isolation in Rust Services

## Principle

Type safety does not automatically provide tenant safety. Multi-tenant backends still need explicit boundary design for authorization, quotas, caching, and resource fairness.

## Isolation Goals

Isolation should protect:
- correctness and authorization boundaries
- latency fairness between tenants
- queue age and worker availability
- shared cache and pool utilization
- downstream dependency blast radius

## Rules

- resolve tenant context at the boundary
- propagate tenant identity explicitly through request-scoped context
- make cache and queue ownership tenant-aware
- define fairness and rate-limit policy
- avoid global shared bottlenecks that let one tenant degrade everyone else

## Bad vs Good

```text
❌ BAD
All tenants share the same concurrency pool and backlog, and operators discover noisy neighbors only after latency spikes hit everyone.

✅ GOOD
High-cost operations have explicit fairness policy, tenant context survives async boundaries, and noisy-neighbor signals are visible before broad impact.
```

## Review Questions

- can one tenant saturate a shared pool?
- are logs and metrics tenant-safe?
- does error handling leak cross-tenant information?
- are background jobs tenant-scoped or global?

## Principal Heuristics

- Global protection without tenant-aware fairness is only partial isolation.
- Perfect isolation is expensive; define where blast-radius control matters most.
- If operators cannot identify a noisy tenant quickly, the system is under-instrumented.
