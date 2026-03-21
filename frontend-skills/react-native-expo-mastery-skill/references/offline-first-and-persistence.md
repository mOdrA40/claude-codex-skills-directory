# Offline-First and Local Persistence

## Rules

- decide what must work offline and what may degrade
- version local persistence schemas intentionally
- avoid hiding sync conflicts from users when business correctness matters
- treat local storage corruption and migration failure as real incidents

## Design Questions

Before adding persistence, decide:

- what data is only a cache
- what data is user-authored and must survive app restart
- what data must be encrypted or protected more strongly
- what data becomes dangerous if stale

## Common Failure Modes

### Cache pretending to be truth

The app renders old data as if it is authoritative, and users only discover the mismatch later.

### Migration damage after update

An app update changes storage format or assumptions, and users lose drafts or hit crashes during startup.

### Conflict denial

The product assumes local and remote state will merge cleanly, but actual user behavior creates edit conflicts and duplicate submissions.

## Recovery Expectations

For important persisted data, define:

- corruption handling
- migration fallback behavior
- user-visible recovery messaging
- telemetry for migration failures and sync conflict spikes

## Review Questions

- what data is authoritative locally?
- how are schema migrations handled on app update?
- what conflict resolution model exists?
- what is the user experience if local data cannot be migrated safely?
