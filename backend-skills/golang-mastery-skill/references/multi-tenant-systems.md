# Multi-Tenant Systems (Go Services)

Multi-tenancy is rarely a schema choice only. It is an isolation, security, operability, and cost decision.

## Isolation Models

## Row-level tenancy

Prefer when:

- tenant count is high,
- cost efficiency matters,
- operational simplicity matters more than hard isolation.

Risks:

- authorization bugs become cross-tenant data leaks,
- noisy tenants can affect others,
- indexing and partitioning strategy matters earlier.

## Schema-per-tenant

Prefer when:

- tenant isolation requirements are stronger,
- onboarding/offboarding workflows are manageable,
- you can automate migrations safely across many schemas.

Risks:

- migration fan-out complexity,
- operational overhead,
- test matrix explosion.

## Database-per-tenant

Prefer when:

- compliance/isolation is the top priority,
- high-value tenants justify cost,
- backup/restore isolation is required.

Risks:

- high ops cost,
- fleet management complexity,
- observability and deployment coordination overhead.

## Non-Negotiable Rules

- Every request must carry explicit tenant context.
- Authn and authz must resolve tenant scope server-side.
- Logs, traces, and metrics must include tenant identifiers where safe.
- Rate limits and quotas should be enforceable per tenant.
- Background jobs must preserve tenant context end-to-end.

## Principal Review Lens

Ask:

- What is the blast radius of one authz bug?
- Can one noisy tenant starve others?
- How are migrations rolled out across tenants?
- Can support/debugging isolate one tenant quickly without exposing another?
