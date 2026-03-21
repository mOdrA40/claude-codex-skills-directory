# Error Boundaries and Resilience in Remix

## Principle

Resilience in Remix depends on route-level failure modeling, not just generic error pages. Nested routes should fail in ways users and operators can understand.

## Common Failure Modes

- one route failure replacing too much UI
- nested route recovery behavior left undefined
- stale or partial data rendered without clear signaling

## Resilience Heuristics

### Fail at the smallest useful route boundary

If one nested segment can fail while preserving the rest of the user journey, the route hierarchy should make that possible.

### Distinguish route failure from mutation failure

A failed mutation should not automatically destroy the route if the user can still recover or continue safely.

### Make degraded states trustworthy

If the route shows stale or partial data, users should be able to understand that they are not seeing fully fresh or complete state.

## Review Questions

- what should fail locally vs globally?
- how does the user recover from route-specific failure?
- which route boundary is carrying too much blast radius?
- what failure is currently forced global only because route boundaries are poorly chosen?
