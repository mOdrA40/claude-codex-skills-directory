# Security and RBAC

## Rules

- RBAC should be least privilege and reviewable by humans.
- Service accounts, secrets, and admission controls are core security boundaries.
- Default-deny posture is healthier than cleanup later.
- Cluster-admin sprawl is a platform failure.

## Principal Review Lens

- Which subject has more privilege than it needs?
- Can workload compromise pivot into cluster compromise?
- Are security controls enforceable during incidents?
