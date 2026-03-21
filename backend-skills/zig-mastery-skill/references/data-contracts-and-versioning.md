# Data Contracts and Versioning in Zig Backends

## Principle

Serialization formats and API payloads are operational contracts. Explicit languages like Zig benefit most when those contracts are treated as stable boundaries instead of incidental structs.

## Rules

- keep wire contracts separate from internal domain types when semantics differ
- prefer additive change first
- define parsing limits and validation clearly
- document compatibility windows for readers and writers

## Bad vs Good

```text
❌ BAD
One struct is used as API payload, DB row, queue event, and internal domain object.

✅ GOOD
Wire contracts are explicit and translated at boundaries.
```

## Review Questions

- which consumers depend on this payload shape?
- can old and new versions coexist?
- where are invalid payloads rejected?
