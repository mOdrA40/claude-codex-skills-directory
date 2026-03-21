# Incidents and Production Debugging in Angular

## Principle

Production Angular incidents usually involve route, state, and async complexity together. Debugging gets easier when ownership is clear before the incident happens.

## Common Incident Classes

- auth and guard loops
- route-specific load regressions
- stream or state leaks
- stale data and invalidation confusion
- enterprise-config-specific environment regressions

## Review Questions

- what changed in route policy, async graph, or environment?
- is the blast radius limited to one feature boundary?
- which logs or metrics would have made this incident easier to classify?
