# Review Checklists by Change Type on the BEAM

## New Endpoint or Consumer

- validate at boundary
- define dependency timeout and degradation behavior
- confirm whether long-lived process is truly needed
- add telemetry for mailbox, lag, and failure signals

## OTP Structure Change

- confirm supervision blast radius
- confirm process ownership
- avoid shared bottlenecks
- bound fan-out and backlog

## Schema or Contract Change

- confirm old and new nodes can coexist
- confirm consumer replay and compatibility posture
- confirm rollback safety
- separate compatibility rollout from cleanup

## Incident Fix

- contain first
- avoid broad rescue blocks
- add missing telemetry if diagnosis was weak
