# Security and Compliance (PostgreSQL)

## Rules

- Apply least privilege per service and workload.
- Protect secrets, backup artifacts, and replication channels.
- Row-level security is powerful but must be verified carefully.
- Audit and masking requirements belong in design, not afterthoughts.

## Security Heuristics

### Database security should match real data gravity

The more valuable or regulated the data, the less acceptable it is to rely on application-only assumptions without reinforcing controls at the database and operational layers.

### Privilege design should stay reviewable

Roles, ownership, replication access, backup handling, and support workflows should remain understandable enough that least privilege can be checked rather than assumed.

### RLS and masking need operational verification

Powerful features like row-level security reduce risk only when teams actively test, audit, and understand how they interact with app behavior and support tooling.

## Common Failure Modes

### App-only trust model

The team assumes the application tier guarantees tenant and data boundaries strongly enough, while database roles and operational workflows remain too broad.

### Privilege sprawl by convenience

Broad service roles or support access paths accumulate over time until least privilege exists mostly on paper.

### Compliance control that fails under incident stress

The official policy looks sound, but one urgent restore, export, or debugging path bypasses it when pressure arrives.

## Principal Review Lens

- Which role can exfiltrate too much data today?
- Are tenant boundaries enforced in schema, app, or both?
- What compliance control will break first under incident pressure?
- Which operational shortcut currently undermines the intended security model most?
