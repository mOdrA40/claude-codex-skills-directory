# API and Event Schema Compatibility Matrix for Rust Services

## Safe-by-Default Changes

- additive optional fields
- backward-compatible response extensions
- event additions that old consumers ignore safely

## Risky Changes

- removing fields used by old binaries or consumers
- tightening validation before all producers are updated
- changing semantic meaning without explicit version strategy
- synchronized rollout assumptions across handlers and workers

## Agent Questions

- can old and new versions coexist?
- is rollback safe after this change?
- are queues, events, and background workers in scope?
