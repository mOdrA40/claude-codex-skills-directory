# Backend Cost and Performance Tradeoffs in Zig Services

## Common Tradeoffs

- tighter memory control can improve efficiency while increasing implementation and review cost
- more buffering can smooth bursts while hiding overload until memory pressure spikes
- extra queues can isolate work while increasing replay and observability burden
- aggressive optimization can improve hot paths while making incidents harder to diagnose

## Agent Questions

- is the bottleneck measured?
- does this help a critical path or only benchmark optics?
- what new failure mode or operational tax does this add?
- is explicit shedding or simpler resource limits better?

## Principal Heuristics

- Low-level control is valuable only if the team can operate it safely.
- Throughput wins that reduce rollback clarity are often too expensive.
