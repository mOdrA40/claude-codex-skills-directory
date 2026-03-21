# Roles, Groups, Authorization, and Policy Design

## Rules

- Role and group design should reflect real access semantics and ownership.
- Authorization models must remain explainable to product and platform teams.
- Group sprawl and composite-role sprawl create long-term risk.
- Identity should not be overloaded to solve every authorization problem poorly.

## Practical Guidance

- Separate authentication identity from business authorization where appropriate.
- Keep admin, support, service, and end-user access clearly distinct.
- Review role naming and scope for least privilege.
- Document how policy changes propagate through clients and teams.

## Principal Review Lens

- Which role has more power than its owner realizes?
- Are we solving access control elegantly or piling up brittle mappings?
- What authorization path is hardest to audit today?
- Which policy simplification would most reduce future risk?
