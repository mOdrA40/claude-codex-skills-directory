# Governance, Admin Security, and Multi-Team Operations

## Rules

- Shared IAM platforms need strict admin boundaries, review paths, and ownership clarity.
- Realm and client admins should not have broader reach than necessary.
- Governance should reduce identity chaos while preserving team autonomy.
- Sensitive realm changes deserve stronger process than ordinary app config.

## Practical Guidance

- Track owners of realms, clients, IdP integrations, and high-risk roles.
- Separate platform-admin and app-admin capabilities where possible.
- Audit admin events and high-risk configuration changes regularly.
- Keep break-glass and emergency admin workflows explicit.

## Principal Review Lens

- Which admin path has the highest blast radius today?
- Are teams over-privileged because governance is weak?
- What ownerless identity config is most dangerous?
- Which control most strengthens IAM platform trust?
