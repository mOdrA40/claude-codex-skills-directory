# Multi-Tenant Kafka

## Rules

- Shared Kafka needs quotas, naming discipline, ACLs, and ownership boundaries.
- One tenant or team should not dominate partitions or retention casually.
- Platform policy should guide topic creation and lifecycle.
- Incident response must preserve tenant separation.

## Principal Review Lens

- Which tenant can create most platform pain today?
- How are quotas enforced and observed?
- What topic or team has the highest blast radius?
