# Multi-Tenant PostgreSQL

## Rules

- Choose row, schema, or database isolation based on blast radius and ops cost.
- Make tenant identity explicit in app and database workflows.
- Noisy-neighbor control requires quota, indexing, and workload visibility.
- Support workflows must preserve isolation and auditability.

## Tenancy Heuristics

### Isolation model is an operating model choice

Row, schema, and database isolation each change blast radius, restore flexibility, migration burden, support workflows, and how much tenant-specific debugging can be done safely.

### Tenant identity must remain explicit end-to-end

It should be obvious in schema, queries, tooling, and support processes how tenant context is enforced and audited.

### Noisy-neighbor control needs more than hope

Without strong observability and workload discipline, one tenant can dominate locks, autovacuum effort, storage, or connection pressure long before teams notice.

## Common Failure Modes

### Cheap isolation with expensive incidents

The chosen tenant model saves operational effort initially, then creates painful blast radius or audit complexity when something goes wrong.

### Tenant identity ambiguity

The application "knows" the tenant, but database workflows and support tools do not make that context explicit enough for safe operations.

### Restore path mismatch

The tenancy model looks acceptable until one tenant-specific restore, export, or support request reveals that recovery granularity is much worse than desired.

## Principal Review Lens

- What is the blast radius of one authz mistake?
- Can one tenant dominate storage, locks, or autovacuum effort?
- How are tenant-specific restore or export operations handled?
- Which tenant-isolation shortcut is most likely to become unacceptable at larger scale or stricter compliance?
