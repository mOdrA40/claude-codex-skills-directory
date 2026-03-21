# Secrets, Permissions, and Supply-Chain Safety

## Rules

- Repository automation must default to least privilege.
- Token scopes, environment protections, and secret exposure paths should be explicit.
- Third-party actions and artifact trust are part of software supply-chain risk.
- Pull request context from forks or untrusted actors needs special care.

## Practical Guidance

- Separate read-only CI from privileged release or infra workflows.
- Use environment approvals and secret scoping intentionally.
- Pin actions and review provenance where possible.
- Minimize long-lived credentials in favor of federated or short-lived auth when supported.

## Principal Review Lens

- Which workflow can exfiltrate too much today?
- Are we trusting more third-party code in CI than we admit?
- What permission is broader than its owning team realizes?
- Which secret path most needs redesign?
