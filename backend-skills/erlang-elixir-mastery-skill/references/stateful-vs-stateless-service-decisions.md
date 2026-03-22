# Stateful vs Stateless Service Decisions on the BEAM

## Stateless Fits When

- request/response dominates
- horizontal scaling should stay simple
- durable state belongs in data infrastructure

## Stateful Fits When

- process state is part of correctness or coordination
- local state materially improves latency or supervision semantics

## Agent Questions

- what happens on restart, failover, or netsplit?
- is process state essential or convenience?
- does statefulness complicate rollout, fairness, or recovery?
