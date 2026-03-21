# Security, Tenancy, and Governance

## Rules

- Access control, tenant boundaries, and data lifecycle must be explicit.
- Shared clusters need strong naming, index ownership, and retention discipline.
- Sensitive data requires auditability and careful query permissions.
- Governance should prevent chaos without freezing useful search workflows.

## Governance Guidance

- Define who owns templates, ingest pipelines, index patterns, and retention rules.
- Keep high-risk admin operations controlled and reviewable.
- Standardize tenant or team isolation patterns.
- Ensure incident shortcuts do not violate data boundaries.

## Principal Review Lens

- Which team can see or change too much today?
- What tenant boundary is most likely to erode under pressure?
- Are governance controls aligned with real risk or just process?
- What policy change would most improve platform safety?
