# Operational Smells and Red Flags in Node.js Services

## Purpose

This guide helps agents detect design and production smells before they become incidents.

## Red Flags

- unbounded retries across multiple layers
- no timeout on outbound HTTP or database calls
- request handlers doing heavy fan-out inline
- background jobs with no owner, lag metric, or dead-letter posture
- event loop delay ignored while adding more traffic
- one tenant or route dominating shared concurrency
- rollout requiring every instance to switch at once

## Agent Review Questions

- what fails first under 10x latency on a dependency?
- where is overload absorbed: queue, memory, event loop, or DB pool?
- does this change increase blast radius across tenants or routes?
- can operators distinguish deploy regression from dependency failure quickly?

## Principal Heuristics

- If backpressure is implicit, failure mode is uncontrolled.
- If retries are not budgeted, they are likely amplification.
- If observability cannot answer what changed, rollback decisions become guesswork.
