# Survivability and Failures (CockroachDB)

## Rules

- Regional/node failure should be an exercised scenario, not a hopeful one.
- Distinguish degraded latency from correctness failure.
- Recovery plans must include application behavior under retry pressure.
- Measure recovery time and blast radius per topology.

## Failure Heuristics

### Multi-region survivability is about user paths, not just quorum math

A topology can satisfy formal survivability goals while still creating unacceptable latency, retry pressure, or degraded user behavior for the business journeys that matter most.

### Slow region scenarios matter almost as much as dead region scenarios

Many painful incidents come from impairment, not total loss. Runbooks and tests should cover slow, unstable, or partially degraded regions explicitly.

### Application retry behavior is part of recovery

The database may remain correct while the application creates user-visible pain through retry storms, duplicated effort, or opaque failure handling.

## Common Failure Modes

### Quorum comfort, poor user recovery

Teams feel safe because the database stays up, but critical paths degrade badly and support burden spikes.

### Failure-mode asymmetry blindness

One region failing is tested, but one region slowing down or becoming unstable creates a much messier and less understood user experience.

### Topology promise without operational rehearsal

The design claims survivability that operators and application teams have never practiced under realistic traffic behavior.

## Principal Review Lens

- What breaks first under region impairment?
- Which workloads degrade gracefully and which fail hard?
- Are operators trained on likely failure sequences?
- What failure mode is technically tolerated by topology but still operationally unacceptable?
