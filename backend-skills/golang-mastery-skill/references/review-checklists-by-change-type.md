# Review Checklists by Change Type for Go Services

## New Endpoint or RPC

- validate input at boundary
- pass and honor `context.Context`
- define timeout posture
- map errors consistently
- add observability hooks

## Concurrency Change

- confirm goroutine ownership
- confirm shutdown and cancellation
- bound fan-out and queue depth
- expose lag and saturation signals

## Schema or Contract Change

- confirm mixed-version compatibility
- confirm rollback safety
- confirm consumer tolerance to both shapes
- separate migration from cleanup

## Incident Fix

- contain first
- avoid blind retries
- verify root cause signals are improved
