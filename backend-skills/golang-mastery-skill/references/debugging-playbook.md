# Debugging Playbook for Go Backends

## Goal

Good debugging guidance should narrow likely failure classes quickly.

## Heuristics

### Latency rises, CPU low
Likely dependency slowness, lock contention, or queueing.

### CPU high, latency high
Likely hot loops, excessive serialization, or too much work on the request path.

### Memory rises over time
Investigate retained buffers, queue growth, cache expansion, or goroutine leakage.

### Goroutine count spikes
Look for fan-out storms, leaked workers, or stuck downstream calls.

## Questions

- which dependency or code path changed first?
- is the failure local or systemic?
- is this a release regression, workload shift, or dependency issue?
