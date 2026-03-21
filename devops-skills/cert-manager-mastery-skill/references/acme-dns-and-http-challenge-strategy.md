# ACME, DNS, and HTTP Challenge Strategy

## Rules

- Challenge strategy should follow domain ownership, ingress architecture, and operational control.
- DNS automation is powerful but can widen trust if poorly scoped.
- HTTP challenge paths should be understood relative to ingress and edge topology.
- Certificate issuance reliability depends on external dependencies too.

## Practical Guidance

- Match challenge method to environment, network reachability, and domain governance.
- Keep DNS credentials and provider permissions tightly scoped.
- Test issuance and renewal under realistic ingress conditions.
- Document which teams own which domain and challenge paths.

## Principal Review Lens

- Which challenge path is most fragile today?
- Are DNS permissions broader than necessary?
- What dependency outage would block the most renewals?
- Which challenge choice best matches operational reality?
