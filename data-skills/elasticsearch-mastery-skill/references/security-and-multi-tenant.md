# Security and Multi-Tenant (Elasticsearch)

## Rules

- Access control, index boundaries, and tenant isolation must be explicit.
- Shared clusters need strong naming, role, and retention discipline.
- Sensitive search data needs auditing and masking strategies.
- Operational shortcuts must not bypass tenant boundaries.

## Principal Review Lens

- Which tenant or team can query too broadly today?
- Are index naming and role patterns enforcing ownership clearly?
- What incident action risks data exposure?
