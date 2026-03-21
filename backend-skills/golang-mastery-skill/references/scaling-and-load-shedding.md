# Scaling and Load Shedding in Go Services

## Principle

Scaling is not only adding pods or goroutines. Mature Go services define how they degrade when dependencies slow down, queues grow, or traffic exceeds safe capacity.

## Rules

- define what traffic is critical vs optional
- reject early when overload is safer than queue growth
- keep concurrency and queue depth bounded
- scale only after understanding the bottleneck
- separate CPU, IO, and dependency saturation signals

## Review Questions

- what should the service shed first?
- are retries amplifying overload?
- does autoscaling help the true bottleneck or just spread pain wider?
