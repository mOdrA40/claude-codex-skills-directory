# Reliability and Operations (Vault)

## Operational Defaults

- Monitor auth failures, lease churn, storage health, seal state, replication health, and audit pipeline integrity.
- Keep upgrades, policy changes, and engine enablement staged and reversible.
- Distinguish secret-platform outages from downstream app auth failures quickly.
- Document safe emergency access and recovery workflows explicitly.

## Run-the-System Thinking

- Vault deserves strong SLO thinking when many systems depend on it for startup or rotation.
- Operational maturity includes rehearsed recovery, not only secure design.
- Secret-platform operators should know which paths and engines are most critical.
- Simplicity and disciplined auth/policy design beat feature sprawl.

## Principal Review Lens

- Which Vault dependency failing would hurt most business systems fastest?
- Can the team recover from seal, backend, or policy mistakes safely?
- What operational practice most increases trust in the platform?
- Are we operating a hardened secret system or a fragile critical dependency?
