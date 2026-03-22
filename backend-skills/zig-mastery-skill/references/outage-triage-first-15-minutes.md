# Outage Triage: First 15 Minutes in Zig Services

## First Questions

- was there a release or config change?
- is pressure caused by memory growth, queue age, or dependency slowness?
- is allocator churn or parsing pressure causing local collapse?
- is one endpoint or tenant dominating the workload?

## Containment Options

- stop rollout
- reject low-value traffic early
- shed optional work
- pause or slow workers causing backlog amplification
- protect memory and dependency ceilings before preserving nice-to-have throughput

## Avoid

- treating every issue as a pure code bug before checking workload shape
- adding buffers as a first response
- restarting blindly without identifying the saturating resource
- coupling incident containment with broad redesign

## Agent Heuristic

In Zig backends, identify the resource boundary under stress first: memory, queue depth, or dependency latency.
