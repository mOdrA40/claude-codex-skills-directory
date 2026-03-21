# Backpressure, Scaling, and Performance Tuning

## Rules

- Backpressure is a system signal, not merely a performance nuisance.
- Scaling decisions should reflect operator balance, source/sink limits, and state movement cost.
- Throughput gains that break correctness or recoverability are false wins.
- Tune with realistic burst, skew, and sink behavior.

## Practical Guidance

- Observe operator lag, checkpoint impact, network shuffle, and sink throughput together.
- Identify whether bottlenecks are CPU, state, network, or external I/O.
- Avoid premature parallelism increases that simply move pressure downstream.
- Benchmark under partial failure and rescale events too.

## Principal Review Lens

- Which operator is the real bottleneck right now?
- Are we tuning symptoms or redesigning for sustainable flow?
- What scaling action most risks destabilizing stateful recovery?
- Which workload should be isolated first under severe pressure?
