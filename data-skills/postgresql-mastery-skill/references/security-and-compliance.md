# Security and Compliance (PostgreSQL)

## Rules

- Apply least privilege per service and workload.
- Protect secrets, backup artifacts, and replication channels.
- Row-level security is powerful but must be verified carefully.
- Audit and masking requirements belong in design, not afterthoughts.

## Principal Review Lens

- Which role can exfiltrate too much data today?
- Are tenant boundaries enforced in schema, app, or both?
- What compliance control will break first under incident pressure?
