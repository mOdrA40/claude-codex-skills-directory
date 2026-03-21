# Service Rollouts and Compatibility in Rust

## Principle

A Rust service can be type-safe and still unsafe to deploy. Compatibility, readiness, shutdown, and background work behavior must be defined across versions.

## Rules

- define readiness as traffic-safe, not just process-alive
- keep schema and event compatibility explicit during rollout
- avoid deploys that require every instance to switch at once unless absolutely necessary
- test shutdown and drain behavior for both API and worker paths

## Review Questions

- can old and new versions coexist?
- what happens to in-flight jobs during rollout?
- what signal tells operators to halt or roll back?
