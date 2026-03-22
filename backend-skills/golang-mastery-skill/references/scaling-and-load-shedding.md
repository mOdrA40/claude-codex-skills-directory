# Scaling and Load Shedding in Go Services

## Principle

Scaling is not only adding pods or goroutines. Mature Go services define how they degrade when dependencies slow down, queues grow, or traffic exceeds safe capacity.

## Capacity Questions

Answer these before talking about autoscaling:
- what saturates first: CPU, memory, DB pool, queue age, downstream rate limits, or lock contention?
- which routes or workloads are highest value?
- which work can be dropped, degraded, queued, or rejected?
- what burst is normal versus dangerous?
- can one tenant or one caller class dominate shared resources?

## Rules

- define what traffic is critical vs optional
- reject early when overload is safer than queue growth
- keep concurrency and queue depth bounded
- scale only after understanding the bottleneck
- separate CPU, IO, and dependency saturation signals

## Bad vs Good

```text
❌ BAD
Traffic rises, goroutines multiply, queues grow, DB acquisition slows, retries pile on, and autoscaling simply spreads the same pain to more instances.

✅ GOOD
The service detects overload early, sheds lower-value work first, and protects critical workflows before shared dependencies collapse.
```

## Review Questions

- what should the service shed first?
- are retries amplifying overload?
- does autoscaling help the true bottleneck or just spread pain wider?

## Principal Heuristics

- Protect critical user journeys before preserving total throughput optics.
- Prefer explicit concurrency and queue ceilings over optimistic buffering.
- If overload response begins only after hard saturation, it is already late.
