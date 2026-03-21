# Multi-Tenant Governance and Access Control

## Rules

- Shared trace platforms require explicit tenant and access boundaries.
- Sensitive traces and metadata may require tighter retention and permissions.
- High-volume tenants should not silently dominate platform behavior.
- Governance should support both operational debugging and security control.

## Governance Guidance

- Standardize tenant identity and access review.
- Track which services and teams create highest ingestion and query cost.
- Make exceptions explicit for critical or regulated workloads.
- Keep administrative actions auditable.

## Principal Review Lens

- Which tenant has the highest blast radius today?
- What access path is broader than it should be?
- Are governance controls mapped to real risk?
- Which policy improvement most strengthens platform trust?
