# SSR, Streaming, and Hydration in SolidStart

## Principle

SolidStart is powerful because it can stream and hydrate efficiently, but that power creates bugs when data ownership and runtime assumptions are unclear.

## Rules

- know what data is server-fetched vs client-fetched
- stream intentionally, not by accident
- keep browser-only logic behind mount guards
- preserve deterministic initial render output

## Review Questions

- which content must be available on first stream?
- what can be deferred safely?
- could hydration mismatch occur because of non-deterministic values or browser APIs?
