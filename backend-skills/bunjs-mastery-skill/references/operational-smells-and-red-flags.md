# Operational Smells and Red Flags in Bun Services

## Red Flags

- missing timeout or abort posture on outbound calls
- webhook handling without raw-body verification or idempotency
- queue consumers with no concurrency cap
- deployment-sensitive schema change with no compatibility plan
- optional enrichments consuming budget for critical paths
- low-value traffic not shed during overload

## Agent Review Questions

- which dependency or queue fails first under pressure?
- can one tenant dominate shared resources?
- does this change increase lock, lag, or backlog risk?
- can operators stop rollout with objective signals?

## Principal Heuristics

- Fast runtime does not remove operability debt.
- If the failure mode is discovered only in production, guardrails are too weak.
- Overload policy must protect business-critical paths first.
