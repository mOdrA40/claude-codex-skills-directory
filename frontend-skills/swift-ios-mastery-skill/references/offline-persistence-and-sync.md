# Offline Persistence and Sync for iOS Apps

## Rules

- define offline guarantees intentionally
- version local persistence
- surface sync conflicts clearly when correctness matters
- distinguish local optimistic state from confirmed remote state

## Design Heuristics

### Decide what must survive restart

Examples include:

- drafts
- queued uploads or writes
- session state
- essential cached user context

Not every screen state deserves persistence.

### Treat migrations as release-sensitive

If local persistence evolves without migration discipline, users will discover the issue only after update, when recovery is hardest.

### Make sync visible when correctness matters

If an action is pending, failed, or conflicting, the user should not be tricked into believing it is already confirmed remotely.

## Common Failure Modes

### Local cache masquerading as truth

The UI shows old or locally edited data as if the server has already accepted it.

### Migration surprise after release

An app upgrade changes storage assumptions and breaks startup or silently drops user data.

### Conflict blindness

Concurrent edits or pending writes create mismatches that the product never modeled clearly.

## Review Questions

- what is authoritative locally vs remotely?
- how are pending writes represented after app restart?
- what does the user see when migration or sync fails?
