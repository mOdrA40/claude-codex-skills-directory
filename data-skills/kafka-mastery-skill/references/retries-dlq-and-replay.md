# Retries, DLQ, and Replay (Kafka)

## Rules

- Poison messages require an explicit isolation strategy.
- Replay is a product capability and an operational risk.
- Retry policy must avoid infinite loops and hidden backlog growth.
- DLQs are not graveyards; they need ownership and recovery workflow.

## Recovery Heuristics

### Replay is a business event

Replaying a topic or dead-letter stream can be as risky as a production deploy because it changes load, duplicates side effects, and can invalidate downstream assumptions.

### DLQ ownership must be explicit

If no team owns triage, classification, replay safety, and closure workflow, the DLQ is only deferred uncertainty.

### Retry posture should reflect dependency behavior

The right retry strategy depends on whether failures are transient, systemic, poison-message driven, or caused by downstream rate limits and business constraints.

## Common Failure Modes

### DLQ graveyard culture

Messages are successfully diverted away from the hot path, but no one treats them as unresolved correctness or customer-impact debt.

### Replay optimism

Teams assume replay is safe because Kafka supports it, without validating idempotency, downstream capacity, or external side effects.

### Infinite patience loops

Retries keep workloads busy enough to look active while backlog, latency, and business pain continue growing invisibly.

## Principal Review Lens

- Who owns DLQ triage and replay safety?
- Can replay duplicate side effects or overload dependencies?
- What backlog threshold becomes user-visible pain?
- Which retry or replay policy currently looks safe but would fail badly during a large incident?
