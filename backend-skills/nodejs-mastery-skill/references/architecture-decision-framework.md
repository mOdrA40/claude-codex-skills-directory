# Architecture Decision Framework for Node.js Backends

## Purpose

This guide helps principal-level engineers choose architecture deliberately instead of copying whichever backend pattern is fashionable.

## Decision Axes

Evaluate a Node.js service across these dimensions:

- request shape: CRUD, streaming, workflow orchestration, event-driven, realtime
- workload type: IO-bound, CPU-bound, mixed
- team shape: small expert team, broad team, multiple squads
- dependency shape: one database, many downstream services, queue-heavy
- contract risk: stable public API, internal events, partner webhooks
- failure tolerance: best-effort, business-critical, compliance-critical
- deploy model: long-lived services, serverless, edge, workers

## Choose a Thin Layered Service When

- the problem is mostly CRUD plus modest policy
- one or two dependencies dominate
- the team needs clarity and speed more than abstraction flexibility
- the domain model is not very deep

## Choose Stronger Use-Case Boundaries When

- many side effects must be coordinated
- retries, idempotency, and outbox behavior matter
- the transport surface is large or public
- multiple teams will extend the service

## Keep a Modular Monolith When

- the domain has multiple capabilities but they are still tightly coupled
- you need clear boundaries without operational service sprawl
- the cost of distributed coordination is not yet justified

## Split into Multiple Services Only When

- scaling characteristics truly differ
- blast radius must be isolated operationally
- ownership boundaries are stable and real
- contract and observability maturity already exist

## Bad vs Good

```text
❌ BAD
Split into microservices because the company likes microservices.

✅ GOOD
Choose boundaries based on ownership, scaling needs, failure isolation, and contract maturity.
```

## Principal Review Questions

- What is the simplest architecture that survives the next 12 months?
- Which boundary reduces risk instead of increasing coordination cost?
- What incident becomes easier because of this design choice?
- What new failure modes does this design introduce?
