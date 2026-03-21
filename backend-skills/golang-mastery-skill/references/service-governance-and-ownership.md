# Service Governance and Ownership in Go Backends

## Purpose

Backend quality degrades when service boundaries exist in code but not in ownership, operational responsibility, or change discipline.

## Rules

- define who owns runtime health, schema changes, and incident response
- keep API and consumer ownership visible
- document operational budgets such as latency, retries, and queue lag
- align code boundaries with team responsibility where possible

## Review Questions

- who owns this service during an incident?
- who approves breaking schema or contract changes?
- is on-call reality reflected in architecture decisions?
