# Service Governance and Ownership in Go Backends

## Purpose

Backend quality degrades when service boundaries exist in code but not in ownership, operational responsibility, or change discipline.

## Governance Scope

Ownership should cover more than code review. It should include:
- runtime health and SLO posture
- contract and schema change approval
- incident response and postmortem follow-through
- dependency risk acceptance
- rollout and rollback decision authority

## Rules

- define who owns runtime health, schema changes, and incident response
- keep API and consumer ownership visible
- document operational budgets such as latency, retries, and queue lag
- align code boundaries with team responsibility where possible

## Bad vs Good

```text
❌ BAD
A service has clear packages and clean code, but nobody can say who approves breaking event changes, who owns rollback decisions, or who must fix chronic queue lag.

✅ GOOD
Ownership is explicit for contracts, operations, incidents, and reliability posture, and those responsibilities influence design and release decisions.
```

## Review Questions

- who owns this service during an incident?
- who approves breaking schema or contract changes?
- is on-call reality reflected in architecture decisions?

## Principal Heuristics

- Architecture without ownership becomes operational debt.
- On-call burden should shape service boundaries and dependency choices.
- If incident lessons never change release or design behavior, governance is weak.
