# Testing (Senior+)

## Determinism

- Avoid real time in tests; inject a clock (or abstract time behind an interface).
- Avoid sleeps; use channels, contexts, and explicit synchronization.
- Use `t.Cleanup` for resource cleanup; avoid leak-prone global state.

## Fuzzing (Go built-in)

- Use fuzz tests for parsers, validators, and input-heavy functions.
- Run locally: `go test -fuzz=FuzzName -fuzztime=30s ./path`

## Concurrency tests

- Always include cancellation paths in tests (no goroutine leaks).
- Run race detector for concurrent code: `go test -race ./...`
- Consider `-count=1` to avoid cache when debugging.

## Integration tests

- Spin dependencies with containers (DB/Redis) when needed; keep them stable and time-bounded.
- Test at boundaries (HTTP handlers, DB repos) with realistic inputs and failure simulation.

