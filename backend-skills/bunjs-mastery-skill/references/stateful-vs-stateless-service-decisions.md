# Stateful vs Stateless Service Decisions in Bun Backends

## Stateless Fits When

- request/response dominates
- easy horizontal scaling is desired
- durable state belongs in storage layers

## Stateful Fits When

- realtime coordination or in-memory subscriptions matter
- local state is part of correctness or latency goals

## Agent Questions

- does restart or failover break correctness?
- does statefulness complicate rollout, fairness, or debugging?
- is local state essential or accidental?
