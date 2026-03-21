# Failure Modes and Incident Patterns in Rust Services

## Purpose

Rust removes classes of low-level bugs, but production incidents still happen through queue semantics, compatibility mistakes, dependency collapse, and shutdown bugs.

## Failure Mode: Type-Safe but Operationally Unsafe

The service had:

- typed errors
- clean async abstractions
- no panics on core paths

But it lacked:

- bounded concurrency
- clear retry owner
- rollout compatibility rules
- shutdown tests for worker paths

Result: duplicate work, lag spikes, and operator confusion during deploys.

## Failure Mode: Buffered Safety Illusion

Channels and streams made code look elegant, but buffers were effectively unbounded relative to production load. Memory pressure rose slowly and then suddenly became an outage.

## Lessons

- correctness includes operability
- bounded queues are more important than elegant async graphs
- every background task needs lifecycle, metrics, and drain semantics

## Review Questions

- where could this service degrade silently before failing loudly?
- what assumptions exist about version compatibility and duplicate work?
- what resource can grow without a hard cap?
