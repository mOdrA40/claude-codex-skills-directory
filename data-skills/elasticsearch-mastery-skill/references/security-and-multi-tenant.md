# Security and Multi-Tenant (Elasticsearch)

## Rules

- Access control, index boundaries, and tenant isolation must be explicit.
- Shared clusters need strong naming, role, and retention discipline.
- Sensitive search data needs auditing and masking strategies.
- Operational shortcuts must not bypass tenant boundaries.

## Tenancy Heuristics

### Multi-tenancy is a governance problem before it is a cost optimization

Shared Elasticsearch clusters save resources only if teams can still explain ownership, access boundaries, retention posture, and incident blast radius.

### Index layout and role design should reinforce each other

Tenant boundaries are safer when naming, index strategy, and role policies all point to the same ownership model rather than fighting each other.

### Operational actions need tenant-aware safety

Maintenance, debugging, or emergency tuning should not rely on shortcuts that are acceptable only because the cluster is treated like one giant shared surface.

## Common Failure Modes

### Shared-cluster convenience drift

The platform starts with acceptable shared usage, then gradually accumulates weak role design, broad access, and unclear data ownership.

### Naming without enforcement

Index naming implies structure, but permissions and operational process do not actually enforce the intended tenant boundaries.

### Incident shortcut risk

During pressure, operators use broad queries or actions that solve the moment but expose or affect more tenant data than intended.

## Principal Review Lens

- Which tenant or team can query too broadly today?
- Are index naming and role patterns enforcing ownership clearly?
- What incident action risks data exposure?
- Which shared-cluster convenience is currently creating the biggest governance debt?
