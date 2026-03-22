# Backend Cost and Performance Tradeoffs in Rust Services

## Common Tradeoffs

- more buffering can smooth spikes while worsening tail latency and memory risk
- more concurrency can increase throughput while worsening lock, pool, or allocator contention
- extra retries can improve local success while amplifying dependency outages
- extra queues can isolate workflows while increasing replay and operator burden

## Agent Questions

- is the bottleneck measured?
- does this help a critical workflow or only benchmark optics?
- what new failure mode or cost does this add?
- is a simpler limit, timeout, or shed policy better?

## Principal Heuristics

- Operable simplicity usually beats speculative optimization.
- If rollback and debugging get harder, the optimization may not be worth it.
