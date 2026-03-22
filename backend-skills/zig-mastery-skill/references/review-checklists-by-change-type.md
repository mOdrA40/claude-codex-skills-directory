# Review Checklists by Change Type for Zig Services

## New Endpoint

- validate size and shape at boundary
- confirm allocator ownership
- define timeout posture
- map errors consistently
- add incident-useful observability

## Worker or Thread Change

- confirm owner and stop signal
- confirm queue bound and overload policy
- confirm failure-reporting path
- expose lag and saturation signals

## Schema or Contract Change

- confirm mixed-version compatibility
- confirm replay safety for workers
- confirm rollback safety
- separate compatibility rollout from cleanup

## Incident Fix

- contain first
- avoid buffering hacks
- add missing memory or queue signals if diagnosis was weak
