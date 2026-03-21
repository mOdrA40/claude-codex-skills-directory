# Reliability and Operations for Node.js Services

## Defaults

Every service should define:

- health and readiness
- graceful shutdown
- timeout policy
- retry ownership
- structured logging
- dependency saturation signals
- rollout and rollback behavior

## Shutdown Sequence

1. stop accepting new traffic
2. fail readiness
3. drain in-flight requests
4. stop background consumers
5. close pools and connections
6. exit before orchestration kills the process

## Common Failure Modes

- dependency timeouts
- retry storms
- queue backlog growth
- event-loop delay spikes
- process memory growth
- large payload latency amplification
