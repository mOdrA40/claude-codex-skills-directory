# Service Rollouts and Compatibility in Rust

## Principle

A Rust service can be type-safe and still unsafe to deploy. Compatibility, readiness, shutdown, and background work behavior must be defined across versions.

## Compatibility Surface

Check compatibility across:
- HTTP and gRPC contracts
- queue events and background jobs
- database schema and migration ordering
- cache keys and derived data
- feature flags and partial rollout assumptions

## Rules

- define readiness as traffic-safe, not just process-alive
- keep schema and event compatibility explicit during rollout
- avoid deploys that require every instance to switch at once unless absolutely necessary
- test shutdown and drain behavior for both API and worker paths

## Bad vs Good

```text
❌ BAD
A new version assumes all instances, consumers, and schemas update together.

✅ GOOD
Old and new versions can coexist safely long enough for rollback, canarying, and staggered background consumer updates.
```

## Review Questions

- can old and new versions coexist?
- what happens to in-flight jobs during rollout?
- what signal tells operators to halt or roll back?

## Principal Heuristics

- Backward compatibility buys operational safety.
- Readiness should mean safe to receive traffic, not merely booted.
- If rollback requires schema surgery or queue draining heroics, rollout design is weak.
