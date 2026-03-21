# Networking (Docker)

## Rules

- Network topology should be simple enough to debug quickly.
- Avoid accidental port exposure and ambiguous service discovery.
- Container network assumptions should match runtime environment.
- DNS and startup timing issues must be expected.

## Principal Review Lens

- Which network path fails first in local or prod environments?
- Are ports exposed more broadly than needed?
- Is connectivity policy obvious to the team?
