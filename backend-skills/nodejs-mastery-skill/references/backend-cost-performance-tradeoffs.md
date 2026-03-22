# Backend Cost and Performance Tradeoffs in Node.js Services

## Purpose

This guide helps agents avoid suggesting optimizations that improve one benchmark while worsening reliability, complexity, or cost.

## Common Tradeoffs

- caching can reduce latency while increasing staleness and invalidation complexity
- more retries can improve local success while amplifying downstream cost and outage duration
- extra queues can isolate work while increasing replay, lag, and operability burden
- microservice splits can improve autonomy while increasing network and deployment complexity

## Agent Questions

- is the bottleneck real or assumed?
- does this optimization protect a critical user journey or just improve vanity numbers?
- what new failure mode or operational cost does this add?
- is a simpler limit, timeout, or shed policy better than a more complex design?

## Principal Heuristics

- Prefer cost-aware simplicity over benchmark theater.
- Do not trade away rollback safety for marginal throughput gains.
- If the team cannot operate the optimization, it is not an optimization.
