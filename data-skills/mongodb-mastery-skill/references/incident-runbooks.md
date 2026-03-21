# Incident Runbooks (MongoDB)

## Rules

- Runbooks should cover lag spikes, elections, shard imbalance, slow queries, and storage pressure.
- Stabilize blast radius before deep diagnosis.
- Include operator-safe actions and explicit anti-actions.
- Recovery must be tied to measurable signals.

## Incident Classes

Useful MongoDB runbooks should separate:

- replication lag and election instability
- shard imbalance and hotspot behavior
- slow query or index-regression incidents
- storage and disk pressure
- operator actions that risk making elections, lag, or routing worse

## Operator Heuristics

### Triage by topology first

Before diving into internals, clarify:

- is the issue replica-set local or sharded-cluster wide?
- is one shard, node, or workload class disproportionately affected?
- did failover, routing, or balancing behavior change recently?

### Protect stability before elegance

The first move should reduce user pain and topology instability, not chase perfect root cause while elections or lag continue.

## Common Failure Modes

### Election panic

Teams react to primary movement or lag symptoms without understanding whether their action will worsen elections, catch-up burden, or routing instability.

### Temporary shard calm mistaken for recovery

One shard or node looks healthier briefly, but the underlying skew, lag, or query pressure remains.

## Principal Review Lens

- Can on-call reduce user pain in 10 minutes?
- Which action risks making elections or lag worse?
- What proves recovery versus temporary relief?
- What MongoDB incident still depends too heavily on tribal operator knowledge?
