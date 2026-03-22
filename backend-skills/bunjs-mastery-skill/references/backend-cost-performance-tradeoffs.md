# Backend Cost and Performance Tradeoffs in Bun Services

## Common Tradeoffs

- lower latency via aggressive caching can increase invalidation and consistency cost
- more worker concurrency can improve throughput while worsening DB or queue contention
- extra retries can increase success locally while extending outage pain globally
- more queues can isolate workflows while increasing replay and observability burden

## Agent Questions

- is the optimization for critical user value or benchmark optics?
- what new failure mode does this add?
- does this improve p99 under real dependency pressure or only local tests?
- is a simpler shed or timeout policy better?

## Principal Heuristics

- Throughput without operability is fragile.
- Lower cost and simpler rollback often beat clever tuning.
