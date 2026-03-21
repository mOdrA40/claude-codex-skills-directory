# Multi-Tenant Governance and Security

## Rules

- Shared NATS platforms require account, subject, and permission discipline.
- Tenant isolation must be explicit in both auth and subject policy.
- One team should not be able to create uncontrolled subject sprawl or resource pressure.
- Governance should preserve NATS simplicity while preventing chaos.

## Governance Guidance

- Standardize account usage, subject prefixes, and operational ownership.
- Protect administrative capabilities and system subjects tightly.
- Track which streams and consumers are highest blast radius.
- Make exception paths explicit and reviewable.

## Principal Review Lens

- Which tenant can create the most platform pain today?
- Are subject permissions aligned with real boundaries?
- What governance gap is already causing confusion?
- Which policy would most improve platform safety with least friction?
