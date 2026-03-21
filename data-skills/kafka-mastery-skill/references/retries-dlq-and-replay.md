# Retries, DLQ, and Replay (Kafka)

## Rules

- Poison messages require an explicit isolation strategy.
- Replay is a product capability and an operational risk.
- Retry policy must avoid infinite loops and hidden backlog growth.
- DLQs are not graveyards; they need ownership and recovery workflow.

## Principal Review Lens

- Who owns DLQ triage and replay safety?
- Can replay duplicate side effects or overload dependencies?
- What backlog threshold becomes user-visible pain?
