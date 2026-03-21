# Incident Runbooks (NATS)

## Cover at Minimum

- Connection storm or client reconnect flood.
- Stream lag or redelivery surge.
- Cluster route degradation.
- Bad subject or permission rollout.
- Replay-related downstream overload.
- Node loss or topology impairment.

## Response Rules

- Stabilize critical traffic paths first.
- Prefer targeted isolation over broad topology panic.
- Preserve evidence around lag, route health, and permission changes.
- Communicate clearly when durability or replay guarantees are degraded.

## Principal Review Lens

- Can responders isolate the failing account, stream, or subject quickly?
- Which emergency action risks worse replay pain later?
- What proves the platform is healthy again?
- Are runbooks practical enough for real multi-team incidents?
