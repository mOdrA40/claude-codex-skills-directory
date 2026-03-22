# Backend Cost and Performance Tradeoffs in Go Services

## Common Tradeoffs

- more goroutines can raise throughput while worsening contention and tail latency
- caches can cut latency while increasing inconsistency and invalidation cost
- retries can lift success locally while amplifying dependency outages
- queue isolation can help correctness while increasing operator burden

## Agent Questions

- is the bottleneck measured?
- is this optimization protecting critical flows or vanity metrics?
- what new operational cost appears?
- is a simpler limit or shed rule better than a more complex design?

## Principal Heuristics

- Measured simplicity beats speculative optimization.
- If rollback gets harder, performance wins may not be worth it.
