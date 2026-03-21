# Offline Sync and Local Persistence

## Rules

- define source of truth clearly
- version persistence schemas intentionally
- keep sync and conflict behavior visible when correctness matters

## Source-of-Truth Models

### Local-first

Useful when:

- offline behavior is core to the product
- users create or edit data in unstable network conditions

### Server-authoritative with queued writes

Useful when:

- business correctness is central
- user actions must eventually reconcile with backend truth

### Hybrid model

Useful only when the team can explain clearly which fields or workflows are local-first vs server-first.

## Common Failure Modes

### Persistence without migration strategy

The app stores useful data locally but cannot evolve schema safely across releases.

### Hidden sync backlog

Users think actions completed, but a queue of local changes is still waiting to synchronize.

### Shared abstraction hides platform behavior

KMP code abstracts persistence well enough for happy paths, but platform-specific storage failure modes are not surfaced clearly.

## Review Questions

- what is the source of truth for each critical workflow?
- what happens when the app updates with pending local changes still unsynced?
- how are migration failure and backlog growth exposed operationally?
