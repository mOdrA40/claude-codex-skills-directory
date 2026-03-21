# Multi-Tenant Clusters

## Rules

- Shared clusters need clear namespace, RBAC, quota, and policy boundaries.
- One tenant should not dominate cluster resources silently.
- Platform support workflows must preserve tenant isolation.
- Cluster tenancy model should be boring enough to operate safely.

## Principal Review Lens

- Which tenant has the highest blast radius today?
- Are quotas, policies, and access controls aligned?
- What incident shortcut could violate isolation?
