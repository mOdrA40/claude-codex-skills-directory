# Outage Triage: First 15 Minutes in Rust Services

## First Questions

- was there a deploy, config drift, or dependency incident?
- is pressure showing up in queues, pools, allocators, locks, or downstream timeouts?
- are async retries and buffered channels amplifying failure?
- is the issue isolated or cross-tenant?

## Containment Options

- stop rollout
- fail readiness on bad instances
- shed low-value or expensive work
- pause toxic consumers
- disable optional enrichment or fan-out

## Avoid

- patching with unchecked `unwrap`/panic behavior
- adding buffering to hide pressure
- assuming type safety means deploy safety
- mixing containment with risky refactor

## Agent Heuristic

During incidents, reduce concurrency pain and blast radius first; only then chase elegant permanent fixes.
