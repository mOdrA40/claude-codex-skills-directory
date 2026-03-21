# Auditability, Compliance, and Secret Lifecycle

## Rules

- Secret lifecycle should include creation, use, rotation, revocation, and retirement.
- Audit trails must be protected, usable, and aligned with compliance obligations.
- Secret sprawl is both a governance and security problem.
- Lifecycle maturity matters more than simply having a secret store.

## Practical Guidance

- Track ownership of sensitive paths and long-lived credentials.
- Review who can mint, read, rotate, or revoke critical secrets.
- Align audit log retention with compliance and forensic needs.
- Periodically retire unused roles, engines, and secret paths.

## Principal Review Lens

- Which long-lived secret is the highest risk today?
- Are audit logs strong enough to support a serious investigation?
- What lifecycle step is most weakly owned?
- Which cleanup initiative most reduces risk quickly?
