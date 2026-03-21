# Multi-Tenant Governance and Access Control

## Rules

- Shared Vault platforms require strong tenant, path, and admin boundaries.
- One tenant should not be able to infer or disrupt another tenant's secret workflows.
- Administrative access should be tightly limited and auditable.
- Governance should focus on trust preservation and supportability.

## Governance Guidance

- Standardize naming and path strategy for teams and environments.
- Separate platform-admin actions from tenant self-service where possible.
- Track high-risk engines, policies, and roles.
- Keep exception processes explicit and reviewable.

## Principal Review Lens

- Which tenant has the largest trust blast radius today?
- What admin capability is too broad?
- Are governance controls mapped to real secret risk?
- What policy improvement would most strengthen shared-platform trust?
