# Backend Cost and Performance Tradeoffs on the BEAM

## Common Tradeoffs

- more processes can improve isolation while increasing coordination and mailbox overhead
- more consumers can increase throughput while worsening backlog or dependency saturation
- extra retries can improve local success while causing restart storms and queue amplification
- extra queues can isolate workflows while increasing replay and operator burden

## Agent Questions

- is the bottleneck measured?
- does this help a critical workflow or only throughput optics?
- what new operational cost or failure mode appears?
- is a simpler backpressure or degrade policy better?

## Principal Heuristics

- Concurrency is not free if mailbox and dependency pressure are unmanaged.
- If on-call cost rises more than user value, the optimization is weak.
