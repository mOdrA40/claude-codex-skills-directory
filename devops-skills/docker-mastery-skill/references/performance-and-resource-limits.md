# Performance and Resource Limits

## Rules

- Containers do not remove CPU and memory reality.
- Limits and reservations should reflect workload behavior.
- Benchmark with realistic I/O, filesystem, and startup patterns.
- Avoid resource settings that create noisy-neighbor chaos.

## Principal Review Lens

- What resource saturates first under peak traffic?
- Are limits protecting the host or killing throughput prematurely?
- Which metric predicts imminent throttling or OOM?
