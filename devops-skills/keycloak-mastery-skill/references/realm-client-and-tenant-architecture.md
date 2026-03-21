# Realm, Client, and Tenant Architecture (Keycloak)

## Rules

- Realm and client boundaries should reflect trust, tenancy, and operational ownership.
- One giant realm is often convenience debt, not maturity.
- Client configuration should encode clear auth expectations and blast radius.
- Tenant separation needs to survive both application growth and admin workflows.

## Design Guidance

- Decide when separation belongs at realm, client, or organizational boundary.
- Keep admin and service-client responsibilities explicit.
- Align environment and tenant boundaries with support and compliance needs.
- Avoid auth architecture that only its original designer understands.

## Principal Review Lens

- Which realm boundary is weakest today?
- Are clients configured around product reality or convenience shortcuts?
- What design choice becomes hardest to reverse later?
- Which tenant or app has too much identity coupling with others?
