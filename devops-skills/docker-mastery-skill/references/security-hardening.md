# Security Hardening (Docker)

## Rules

- Run as non-root where possible.
- Minimize capabilities, writable surfaces, and package surface area.
- Scan images, but do not confuse scanning with real hardening.
- Protect registries, credentials, and build pipelines as part of security posture.

## Principal Review Lens

- Which privilege is unnecessary but still present?
- Are we patching for real exposure or vanity score reduction?
- What supply-chain path is easiest to compromise?
