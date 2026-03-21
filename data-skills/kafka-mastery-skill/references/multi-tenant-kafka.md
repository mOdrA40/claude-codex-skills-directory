# Multi-Tenant Kafka

## Rules

- Shared Kafka needs quotas, naming discipline, ACLs, and ownership boundaries.
- One tenant or team should not dominate partitions or retention casually.
- Platform policy should guide topic creation and lifecycle.
- Incident response must preserve tenant separation.

## Tenancy Heuristics

### Multi-tenancy is a governance model, not just a cluster-saving tactic

Shared Kafka works only when teams can explain ownership, quotas, topic lifecycle, retention posture, and incident blast radius clearly enough to operate the platform safely.

### Fairness needs enforcement, not hope

Without quotas and clear lifecycle control, the loudest tenant, highest-retention workload, or fastest-growing topic family will usually capture disproportionate platform budget.

### Tenant-aware incident handling matters

Operations should be able to reduce one tenant's blast radius without casually breaking unrelated teams or exposing data paths too broadly.

## Common Failure Modes

### Shared-cluster politeness model

The platform assumes teams will behave responsibly without enough technical enforcement, and eventually one tenant or domain creates outsized pain.

### Topic lifecycle drift

Topics survive longer than their value, ownership becomes unclear, and platform sprawl grows faster than anyone governs it.

### Quota blindness

Quotas exist weakly or not at all, so the cluster only reveals unfairness once incidents or capacity crises arrive.

## Principal Review Lens

- Which tenant can create most platform pain today?
- How are quotas enforced and observed?
- What topic or team has the highest blast radius?
- Which multi-tenant shortcut is most likely to become a serious governance problem later?
