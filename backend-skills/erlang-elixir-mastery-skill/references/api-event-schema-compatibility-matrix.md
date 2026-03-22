# API and Event Schema Compatibility Matrix on the BEAM

## Safe-by-Default Changes

- additive optional fields
- backward-compatible response extensions
- event additions old consumers ignore safely

## Risky Changes

- removing fields required by old nodes or consumers
- tightening validation before all producers are updated
- changing semantic meaning without compatibility strategy
- assuming synchronized rollout of nodes and consumers

## Agent Questions

- can old and new nodes coexist safely?
- is rollback safe after this change?
- are queues, consumers, and background jobs included in compatibility scope?
