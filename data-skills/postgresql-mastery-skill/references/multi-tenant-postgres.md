# Multi-Tenant PostgreSQL

## Rules

- Choose row, schema, or database isolation based on blast radius and ops cost.
- Make tenant identity explicit in app and database workflows.
- Noisy-neighbor control requires quota, indexing, and workload visibility.
- Support workflows must preserve isolation and auditability.

## Principal Review Lens

- What is the blast radius of one authz mistake?
- Can one tenant dominate storage, locks, or autovacuum effort?
- How are tenant-specific restore or export operations handled?
