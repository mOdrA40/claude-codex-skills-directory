# Multi-Region Failover and Disaster Posture in Rust Services

## Purpose

A Rust service may be locally robust yet still operationally weak during regional failure. Disaster posture requires clarity on traffic movement, replay, shared dependency risk, and failover ownership.

## Core Principle

Regional resilience is not only about standing up another cluster. It is about defining compatibility and correctness when traffic, writes, and background work move under stress.

## Design Questions

- is the system active-active, active-passive, or region-primary?
- which workflows can tolerate replay or duplication?
- what shared dependencies can become cross-region blast-radius amplifiers?
- how do workers and schedulers behave during failover?
- what stale data or stale config risks exist during traffic shift?

## Review Questions

- can old and new regional writers overlap dangerously?
- what prevents double-processing after failover?
- how are queues and background jobs recovered safely?
- can one degraded region poison healthy regions through shared dependencies?
- is regional failover rehearsed and observable?

## Principal Heuristics

- Simpler failover semantics beat clever coordination nobody can operate.
- Disaster posture must include async workflows, not only request routing.
- If failover requires tribal knowledge, it is not ready.
