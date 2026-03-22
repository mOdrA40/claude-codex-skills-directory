# Operational Smells and Red Flags in Zig Services

## Red Flags

- allocator ownership unclear across module boundaries
- queue growth with no explicit overload policy
- unbounded parsing or buffering on untrusted input
- detached workers with no stop or error path
- release-sensitive format or schema changes with no compatibility plan
- one tenant or endpoint dominating memory or concurrency

## Agent Review Questions

- what resource saturates first: memory, queue depth, dependency latency, or parsing cost?
- does this change hide allocator pressure or make it visible?
- can operators distinguish local resource collapse from dependency failure?
- what signal should halt rollout immediately?

## Principal Heuristics

- If ownership is implicit, leaks and incident ambiguity follow.
- If overload policy is absent, memory becomes the queue.
- If rollback is not explicit, release safety is weak.
