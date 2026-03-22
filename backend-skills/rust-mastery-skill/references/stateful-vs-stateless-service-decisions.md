# Stateful vs Stateless Service Decisions in Rust Backends

## Stateless Fits When

- request/response dominates
- easy horizontal scaling matters
- durable state belongs in data infrastructure

## Stateful Fits When

- local coordination or subscriptions are essential
- in-memory state materially improves correctness or latency

## Agent Questions

- what happens on restart or failover?
- is local state essential or convenience?
- does statefulness complicate rollout, fairness, or recovery?
