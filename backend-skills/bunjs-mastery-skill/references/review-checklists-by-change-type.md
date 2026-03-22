# Review Checklists by Change Type for Bun Services

## New Endpoint

- validate request and config boundaries
- define timeout and abort behavior
- map errors consistently
- add tracing or request ID fields
- define size limits and auth posture

## Worker or Queue Change

- confirm shutdown and drain behavior
- confirm retry/dead-letter rules
- confirm duplicate safety
- confirm concurrency and lag visibility

## Schema or Contract Change

- confirm old and new versions coexist
- confirm workers and handlers tolerate both shapes
- confirm rollback is safe
- separate migration from cleanup

## Incident Fix

- reduce blast radius first
- avoid speculative performance changes
- add missing observability if root cause was unclear
