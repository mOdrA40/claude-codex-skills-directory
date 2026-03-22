# Review Checklists by Change Type for Rust Services

## New Endpoint

- validate at boundary
- map typed errors consistently
- define timeouts and cancellation
- add tracing fields
- define idempotency if side effects exist

## Async or Worker Change

- confirm task ownership
- confirm shutdown and drain posture
- cap concurrency and buffers
- expose lag, queue age, and failure metrics

## Schema or Contract Change

- confirm mixed-version compatibility
- confirm replay safety
- confirm rollback without manual repair
- separate migration from cleanup

## Incident Fix

- contain first
- avoid panic or buffering hacks
- add missing signals if diagnosis was weak
