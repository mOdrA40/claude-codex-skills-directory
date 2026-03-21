# Performance Investigation Playbook for Rust Services

## Principle

Rust makes performance mistakes less obvious because code can look efficient while still paying in allocations, cloning, buffering, or synchronization.

## Investigation Order

1. verify the bottleneck with measurement
2. separate CPU, allocation, lock, and IO causes
3. inspect p95/p99, not just averages
4. compare before and after the suspected change

## Common Failure Classes

- clone-heavy code in hot paths
- oversized buffers and payload copies
- lock contention around shared state
- async tasks buffering too much work
- serialization hotspots

## Review Questions

- is this path slower because of ownership choices, buffering, or IO?
- what metric proves the bottleneck?
- would a simpler design remove the bottleneck entirely?
