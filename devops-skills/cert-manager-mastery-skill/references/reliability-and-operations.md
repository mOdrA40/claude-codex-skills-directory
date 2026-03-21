# Reliability and Operations (cert-manager)

## Operational Defaults

- Monitor renewal health, issuer availability, challenge failure rates, expiry windows, and secret update behavior.
- Keep issuer and policy changes staged and reversible.
- Distinguish cluster automation issues from external DNS/CA dependency failures quickly.
- Document emergency renewal and manual fallback paths for critical certs.

## Run-the-System Thinking

- Certificate automation is critical platform infrastructure once many workloads depend on it.
- Expiry risk should have stronger visibility than routine success metrics.
- Shared issuers and wildcard certs carry high blast radius.
- Operational trust comes from rehearsed renewal and strong governance.

## Principal Review Lens

- Which failure blocks the most certificate renewals fastest?
- Can the team recover critical cert availability safely?
- What operational habit most improves trust in cert automation?
- Are we operating disciplined PKI automation or hoping before expiry?
