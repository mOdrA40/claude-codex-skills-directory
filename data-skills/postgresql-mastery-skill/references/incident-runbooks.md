# Incident Runbooks (PostgreSQL)

## Rules

- Runbooks should start from symptoms: saturation, lag, lock storm, disk pressure, failover.
- Include safe first actions, not only ultimate fixes.
- Protect data integrity before chasing vanity recovery speed.
- Capture rollback and escalation conditions.

## Principal Review Lens

- Can an on-call engineer stabilize the system in 10 minutes?
- What action is explicitly forbidden during the incident?
- Which metric confirms recovery versus temporary relief?
