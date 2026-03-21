# Rollouts and Release Safety for Zig Services

## Purpose

Release safety is part of backend design. A Zig service can be memory-safe at runtime and still be operationally unsafe during rollout if readiness, compatibility, and drain behavior are undefined.

## Rules

- readiness must mean traffic-safe, not merely process-alive
- old and new versions must be checked for config and protocol compatibility
- shutdown behavior must be tested, not assumed
- rollbacks need a defined trigger and a defined safe state

## Common Failure Modes

- new binary starts before dependencies are usable
- old and new instances disagree on payload or schema
- in-flight work is dropped on shutdown
- background workers double-run or abandon work during rollout

## Review Questions

- what marks this instance safe for traffic?
- can two adjacent versions coexist during rollout?
- what is the operator signal to roll back?
