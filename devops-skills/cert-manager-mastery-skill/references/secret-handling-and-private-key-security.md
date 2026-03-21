# Secret Handling and Private-Key Security

## Rules

- Private keys are high-risk assets and require disciplined handling.
- Secret creation, storage, and access should reflect real trust boundaries.
- Key material should not be exposed to more workloads or humans than necessary.
- Automation convenience should not weaken key protection.

## Practical Guidance

- Restrict secret access paths tightly.
- Review which controllers, namespaces, and workloads can read generated secrets.
- Align secret backup and restore with security requirements.
- Document how compromised or leaked keys are handled operationally.

## Principal Review Lens

- Which private key path is weakest today?
- Are we treating key secrets like ordinary Kubernetes data?
- What access boundary most needs tightening?
- Which security improvement most reduces trust risk?
