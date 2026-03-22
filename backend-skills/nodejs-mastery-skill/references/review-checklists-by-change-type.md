# Review Checklists by Change Type for Node.js Services

## New Endpoint

- validate input at the edge
- define error mapping
- add timeout posture for dependencies
- add request correlation fields
- define idempotency if side effects exist

## Queue or Worker Change

- define retry vs dead-letter behavior
- confirm duplicate safety
- confirm shutdown and drain behavior
- expose lag, age, and failure metrics

## Schema or Contract Change

- confirm mixed-version compatibility
- confirm rollback safety
- confirm readers and workers tolerate both shapes
- separate migration from cleanup if possible

## Incident Fix

- contain first, refactor second
- confirm blast radius reduced
- avoid adding blind retries
- add missing signals if diagnosis was weak
