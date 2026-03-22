# API and Event Schema Compatibility Matrix for Bun Services

## Safe-by-Default Changes

- add optional fields
- allow older payload variants during transition
- extend responses in backward-compatible ways

## Risky Changes

- removing fields used by old workers or clients
- tightening validation before all producers are updated
- changing webhook or event meaning without version strategy
- requiring synchronized rollout across handlers and consumers

## Agent Questions

- can old and new versions coexist?
- is rollback still safe?
- are workers, queues, and webhooks included in compatibility scope?
