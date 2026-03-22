# Outage Triage: First 15 Minutes in Bun Services

## First Questions

- was there a recent deploy or config change?
- is the bottleneck app runtime, DB pool, queue lag, or dependency timeout?
- are optional features consuming the budget of critical routes?
- is one tenant or endpoint dominating concurrency?

## First Actions

- pause rollout if signal aligns
- shed low-value traffic
- reduce expensive fan-out or webhook work
- drain broken instances
- protect DB and queue health before preserving secondary features

## Avoid

- blanket retries
- huge logging spikes that worsen IO pressure
- mixing incident containment with risky refactors
- assuming autoscaling alone is a fix

## Agent Heuristic

During outage response, prefer containment steps that reduce blast radius over clever code changes.
