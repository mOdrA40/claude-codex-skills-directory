# Multi-Tenant and Resource Isolation in Rust Services

## Principle

Type safety does not automatically provide tenant safety. Multi-tenant backends still need explicit boundary design for authorization, quotas, caching, and resource fairness.

## Rules

- resolve tenant context at the boundary
- propagate tenant identity explicitly through request-scoped context
- make cache and queue ownership tenant-aware
- define fairness and rate-limit policy
- avoid global shared bottlenecks that let one tenant degrade everyone else

## Review Questions

- can one tenant saturate a shared pool?
- are logs and metrics tenant-safe?
- does error handling leak cross-tenant information?
- are background jobs tenant-scoped or global?
