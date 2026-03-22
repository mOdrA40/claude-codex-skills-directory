# API and Event Schema Compatibility Matrix for Zig Services

## Safe-by-Default Changes

- additive optional fields
- backward-compatible response extensions
- new event fields tolerated by old readers safely

## Risky Changes

- removing fields required by old binaries
- tightening parsing or validation before all producers are updated
- changing semantic meaning without compatibility plan
- assuming synchronized rollout of readers and workers

## Agent Questions

- can old and new binaries coexist?
- is rollback safe after this change?
- are workers, queues, and file or FFI boundaries included in compatibility thinking?
