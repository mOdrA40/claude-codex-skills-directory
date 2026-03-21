# Incident Response Playbook for Bun Services

## Goal

When a Bun service is failing, operators need concrete decision paths, not generic advice.

## Common Incident Classes

- dependency timeout surge
- rollout regression
- queue backlog or stuck consumers
- websocket fan-out overload
- rate-limit misconfiguration

## First Questions

- is the issue request-path, background-path, or both?
- is the current release correlated with the regression?
- is one tenant or feature driving the load?
- should optional features be shed before rollback?

## Immediate Controls

- fail readiness to stop new traffic
- disable optional enrichment paths
- reduce consumer concurrency if dependencies are collapsing
- separate high-cost traffic from critical core requests
