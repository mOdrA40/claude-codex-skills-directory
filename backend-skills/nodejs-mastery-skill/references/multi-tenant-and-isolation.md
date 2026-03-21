# Multi-Tenant and Isolation Patterns in Node.js Services

## Purpose

Multi-tenant systems fail in production not only through security bugs, but also through noisy-neighbor behavior, poor isolation, retry amplification, and data access confusion. Node.js services often hide these risks because handlers look simple while tenant scoping leaks across layers.

## Isolation Questions

Before implementing multi-tenancy, define:

- where tenant identity is established
- how tenant scope is propagated
- whether data isolation is row-level, schema-level, or database-level
- whether cache keys include tenant identity
- whether queues, rate limits, and budgets are tenant-aware
- what the blast radius is if one tenant overloads the system

## Boundary Rule

Tenant resolution belongs at the boundary.

Transport should:

- authenticate caller identity
- resolve tenant context
- validate tenant access rights
- pass a normalized tenant context inward

Domain code should not parse headers to rediscover tenancy.

## Bad vs Good

```typescript
// ❌ BAD: repositories accept raw headers or infer tenant implicitly.
const orders = await repo.listOrders(req.headers['x-tenant-id'])
```

```typescript
// ✅ GOOD: tenant context is normalized once at the edge.
const tenant = requestContext.tenant
const orders = await repo.listOrders({ tenantId: tenant.id })
```

## Cache and Queue Rules

- cache keys must include tenant scope where data is tenant-bound
- queue workloads should identify tenant ownership
- rate limits should distinguish one abusive tenant from the whole platform
- expensive per-tenant jobs need quotas and explicit scheduling policy

## Principal Review Questions

- Can one tenant starve others?
- Can tenant data leak via cache, logs, or metrics labels?
- Which dependencies are tenant-aware vs tenant-blind?
- What happens when one tenant triggers a retry storm?
