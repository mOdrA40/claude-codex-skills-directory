# API and Event Schema Compatibility Matrix for Go Services

## Safe-by-Default Changes

- additive optional fields
- backward-compatible response extensions
- events with added fields ignored safely by old consumers

## Risky Changes

- removing fields used by old binaries
- tightening validation before producers are updated
- changing semantic meaning without version discipline
- requiring synchronized rollout of handlers and consumers

## Agent Questions

- can old and new binaries coexist?
- is rollback safe after this change?
- are queues, gRPC, and async consumers included in compatibility scope?
