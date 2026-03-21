# Failure Stories and Recovery Patterns on the BEAM

## Purpose

The BEAM gives strong recovery primitives, but real incidents still happen when ownership, demand, or distributed assumptions are wrong.

## Failure Story: The Helpful GenServer

A single GenServer slowly became the coordination point for:

- request orchestration
- cache refresh
- async job dispatch
- external API retries

Under load it became a hidden bottleneck.

### Lesson

A process boundary is not automatically a good boundary.

## Failure Story: The Stable Cluster That Wasn't

The team assumed cluster visibility was stable enough for leadership-sensitive jobs. A netsplit produced duplicate work and inconsistent recovery.

### Lesson

If a job must be single-owner, that ownership must be explicit and observable.

## Recovery Patterns

- reduce blast radius first
- isolate overloaded process families
- suspend non-critical consumers
- degrade optional features before core workflows
- prefer explicit leadership or leases over hopeful cluster assumptions

## Review Questions

- where is hidden coordination concentrated?
- what duplicate work can occur during cluster instability?
- what is the fastest safe blast-radius reduction step?
