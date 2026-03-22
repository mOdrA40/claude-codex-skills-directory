# Zero-Downtime Schema and Contract Migrations in Zig Services

## Core Principle

Explicit systems languages still need compatibility discipline. Assume mixed-version traffic and staggered worker rollout.

## Safe Rules

- prefer additive schema and contract changes first
- keep old and new readers compatible during transition
- separate backfill and cleanup from compatibility rollout
- define rollback before release
- treat queues, files, and FFI boundaries as compatibility surfaces when relevant

## Agent Checklist

- can old and new binaries coexist?
- are parsing and serialization paths tolerant enough during migration?
- do workers safely handle replay or duplicate inputs?
- is any cleanup step irreversible too early?
- where will operators detect migration-induced pressure?
