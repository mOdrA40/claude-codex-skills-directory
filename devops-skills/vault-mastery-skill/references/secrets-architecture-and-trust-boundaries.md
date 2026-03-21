# Secrets Architecture and Trust Boundaries (Vault)

## Rules

- Secrets architecture should follow identity, blast radius, and lifecycle, not convenience alone.
- Vault is a control point for sensitive truth and should be modeled accordingly.
- Separate human, machine, platform, and emergency access paths clearly.
- Avoid turning a secure secret system into an unmanaged configuration store.

## Design Guidance

- Define which secrets belong in Vault and which should not.
- Align engine choice with use case, auditability, and rotation expectations.
- Make trust boundaries explicit across tenants, environments, and operators.
- Keep secret retrieval workflows understandable to service owners.

## Principal Review Lens

- Which trust boundary is weakest today?
- Are we storing data in Vault because it is sensitive or because it is convenient?
- What secret workflow creates the most hidden blast radius?
- What architectural cleanup most improves security posture?
