# Topology Design

## Rules

- Exchanges, queues, and bindings should model business routing explicitly.
- Topology sprawl becomes an operational and debugging tax.
- Vhosts and naming conventions should encode ownership.
- Keep dead-letter and retry topology understandable to humans.

## Topology Heuristics

### Topology is a routing contract

Good RabbitMQ topology makes it clear how messages flow, where ownership sits, what the retry / DLQ path is, and which bindings create the largest blast radius.

### Simplicity usually beats clever routing

A more expressive exchange/binding layout is not automatically better if operators and developers can no longer reason about message flow quickly.

### Retry topology must remain legible

Dead-letter, backoff, and retry chains should be understandable enough that humans can predict what happens to a failing message without deep archaeology.

## Common Failure Modes

### Topology sprawl without ownership

Exchanges and bindings multiply faster than governance, making incident diagnosis and safe change harder over time.

### Routing cleverness over operability

The topology is flexible but too complex for engineers and on-call responders to reason about confidently.

### Retry-path obscurity

Messages loop through dead-letter and retry routes in ways the team cannot explain quickly during incidents.

## Principal Review Lens

- Can on-call reason about message flow quickly?
- Which exchange or binding has the biggest blast radius?
- Are we building topology around transient implementation details?
- Which routing path is currently most dangerous because too few people can explain it end-to-end?
