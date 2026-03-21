# Zig Anti-Patterns and Operational Smells

## Purpose

Zig rewards explicit engineering, but explicit syntax does not guarantee explicit architecture. This guide captures backend smells that remain dangerous even in a systems-conscious codebase.

## Smells

### One Struct for Everything

A single struct is used for:

- API payload
- domain state
- database row
- queue message

This creates accidental coupling and unsafe change propagation.

### Explicit but Unowned Concurrency

Threads are spawned explicitly, but no owner, stop condition, or lag metric exists.

### Allocator Discipline Without Runtime Discipline

Memory ownership is documented, but retries, timeouts, and deploy behavior are still vague.

## Lesson

Low-level explicitness must extend to operational behavior or the service is only half explicit.

## Review Questions

- what is explicit in syntax but still implicit in operations?
- what boundary change would create hidden compatibility risk?
- where is the service still relying on “it will probably be fine”?
