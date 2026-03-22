# Stateful vs Stateless Service Decisions in Node.js Backends

## Stateless Fits When

- work is request/response oriented
- horizontal scaling should stay simple
- state belongs in DB, cache, or queue infrastructure

## Stateful Fits When

- in-memory coordination or subscriptions matter
- latency requires carefully managed local state
- externalizing all state would make the system worse

## Agent Questions

- is local state part of correctness or just a convenience cache?
- what happens on restart or failover?
- does statefulness complicate rollout and fairness?

## Principal Heuristics

- Prefer stateless by default.
- Introduce state only when the operational and correctness tradeoffs are worth it.
