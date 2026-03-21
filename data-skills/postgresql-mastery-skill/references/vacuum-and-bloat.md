# Vacuum and Bloat (PostgreSQL)

## Rules

- Autovacuum posture is part of application design on write-heavy systems.
- Bloat, dead tuples, and freeze risk should be measured early.
- Hot tables deserve explicit monitoring and tuning.
- UPDATE-heavy patterns can become silent IO taxes.

## Principal Review Lens

- Which tables are accumulating dead tuples fastest?
- Is autovacuum keeping up with write pressure?
- Are we using the database in a way that guarantees bloat pain?
