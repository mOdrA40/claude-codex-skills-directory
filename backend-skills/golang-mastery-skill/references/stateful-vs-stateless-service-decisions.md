# Stateful vs Stateless Service Decisions in Go Backends

## Stateless Fits When

- request/response dominates
- easy horizontal scaling matters
- state belongs in DB, cache, or queue layers

## Stateful Fits When

- in-memory coordination or subscription state is essential
- local state materially improves correctness or latency

## Agent Questions

- what happens on restart or failover?
- is local state essential or convenience?
- does statefulness complicate rollout, fairness, or recovery?
