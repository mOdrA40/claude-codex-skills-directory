# Incident Runbooks (MongoDB)

## Rules

- Runbooks should cover lag spikes, elections, shard imbalance, slow queries, and storage pressure.
- Stabilize blast radius before deep diagnosis.
- Include operator-safe actions and explicit anti-actions.
- Recovery must be tied to measurable signals.

## Principal Review Lens

- Can on-call reduce user pain in 10 minutes?
- Which action risks making elections or lag worse?
- What proves recovery versus temporary relief?
