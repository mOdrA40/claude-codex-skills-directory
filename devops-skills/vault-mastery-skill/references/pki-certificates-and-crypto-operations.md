# PKI, Certificates, and Crypto Operations

## Rules

- PKI workflows require clear ownership of issuance, trust, rotation, and revocation.
- Certificate automation is powerful but unforgiving when assumptions are wrong.
- Crypto operations should remain auditable and boring.
- Certificate TTL and trust-chain design must match operational maturity.

## Practical Guidance

- Define which workloads and humans trust which CA paths.
- Test rotation and renewal workflows before production dependence.
- Keep private key handling tightly controlled.
- Make revocation and compromise workflows explicit.

## Principal Review Lens

- Which certificate workflow is least rehearsed today?
- What trust assumption would fail hardest during compromise?
- Are crypto operations concentrated in a path too few people understand?
- What PKI design choice most threatens operational stability?
