# Topology Design

## Rules

- Exchanges, queues, and bindings should model business routing explicitly.
- Topology sprawl becomes an operational and debugging tax.
- Vhosts and naming conventions should encode ownership.
- Keep dead-letter and retry topology understandable to humans.

## Principal Review Lens

- Can on-call reason about message flow quickly?
- Which exchange or binding has the biggest blast radius?
- Are we building topology around transient implementation details?
