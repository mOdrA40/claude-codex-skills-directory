# Performance (Measure, Don’t Guess)

## What to measure first

- p50/p95/p99 latency by endpoint and dependency
- CPU profile (pprof)
- memory allocations and heap growth
- lock contention (mutex profile) and blocking (block profile)

## Tooling

- Micro: `go test -bench . -benchmem ./...`
- CPU/mem: `pprof` (for services, expose `/debug/pprof` behind auth)
- Trace: `go tool trace` for scheduler/GC and concurrency bottlenecks

## High-impact Go patterns

- Avoid accidental allocations in hot paths (measure with `-benchmem` and `testing.AllocsPerRun`).
- Avoid `defer` inside very hot loops (measure; sometimes it matters, often it doesn’t).
- Preallocate slices/maps when size is known and it’s a real hotspot.
- Don’t use `sync.Pool` unless you’ve proven allocator pressure and you understand lifecycle semantics.

## Concurrency performance

- Prefer fewer goroutines with bounded work queues over “goroutine per request” for heavy work.
- Keep critical sections small; use RWMutex only when read-heavy and contention is proven.

