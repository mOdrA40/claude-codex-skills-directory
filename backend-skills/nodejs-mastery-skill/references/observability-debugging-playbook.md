# Observability and Debugging Playbook for Node.js

## Goal

You want operators to answer whether a failure is caused by event-loop pressure, dependency slowness, queue lag, or rollout regression in minutes, not hours.

## Must-Have Signals

- request latency and status by route class
- event-loop delay
- dependency latency by downstream name
- queue depth and job age
- process memory and restart behavior
- release version and commit identity

## Debugging Heuristics

### High latency, low CPU
Likely dependency slowness or blocked downstream resources.

### High latency, high event-loop delay
Likely CPU, sync work, excessive serialization, or overloaded logging/compression.

### Rising memory without rising traffic
Investigate queue backlog, buffering, caching, or leaked object retention.

## Incident Questions

- did latency rise before errors or after?
- is one route class dominant?
- which downstream changed behavior first?
- is the new release correlated with the regression?
