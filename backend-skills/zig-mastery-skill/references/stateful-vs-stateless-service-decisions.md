# Stateful vs Stateless Service Decisions in Zig Backends

## Stateless Fits When

- request/response dominates
- horizontal scaling should stay simple
- durable state belongs in data infrastructure

## Stateful Fits When

- local coordination or in-memory state is part of correctness
- latency goals justify carefully controlled local state

## Agent Questions

- what happens on restart or failover?
- is local state essential or convenience?
- does statefulness complicate rollout, fairness, or recovery?
