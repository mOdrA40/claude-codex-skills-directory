# Reliability (SRE-minded Defaults)

## Timeouts & Budgets

- Treat every dependency call as “must finish within a budget”.
- Derive per-hop deadlines from the overall request deadline (don’t stack independent timeouts blindly).

Bad (stacking timeouts without a global budget):

```go
// ❌ BAD: each layer adds its own timeout; total can exceed your SLO
ctx1, c1 := context.WithTimeout(ctx, 10*time.Second)
defer c1()
_ = svc.Call(ctx1)
```

Good (derive per-hop from parent deadline):

```go
// ✅ GOOD: use parent deadline when present
func withBudget(ctx context.Context, max time.Duration) (context.Context, context.CancelFunc) {
	if dl, ok := ctx.Deadline(); ok {
		remaining := time.Until(dl)
		if remaining <= 0 {
			return context.WithCancel(ctx)
		}
		if remaining < max {
			return context.WithTimeout(ctx, remaining)
		}
	}
	return context.WithTimeout(ctx, max)
}
```

## Retries (only when safe)

- Retry only idempotent operations (or make them idempotent with idempotency keys).
- Use exponential backoff + jitter, bounded by a max and an overall deadline.
- Never retry forever; never retry after the deadline.

Bad (retry storm + ignores cancellation):

```go
// ❌ BAD: infinite retry, no jitter, no deadline awareness
for {
	if err := call(); err == nil {
		return nil
	}
	time.Sleep(100 * time.Millisecond)
}
```

Good (bounded retries + jitter + respects context). Prefer a per-service RNG (avoid global `math/rand` in hot paths):

```go
// ✅ GOOD: bounded attempts, exponential backoff, jitter, ctx-aware
func retry(ctx context.Context, attempts int, base, max time.Duration, rng *rand.Rand, fn func(context.Context) error) error {
	var last error
	delay := base
	for i := 0; i < attempts; i++ {
		if err := fn(ctx); err == nil {
			return nil
		} else {
			last = err
		}

		if i == attempts-1 {
			break
		}
		jitter := time.Duration(rng.Int63n(int64(delay / 2)))
		sleep := delay/2 + jitter
		if sleep > max {
			sleep = max
		}
		t := time.NewTimer(sleep)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
		if delay < max {
			delay *= 2
			if delay > max {
				delay = max
			}
		}
	}
	return last
}
```

## Circuit breakers (stop retry storms)

- When a dependency is failing, “try harder” makes it worse.
- Add circuit breakers for unstable downstreams and fail fast while open.
- Combine with bulkheads so one dependency can’t starve the service.

Minimal breaker sketch (good enough for many internal calls):

```go
// ✅ Minimal circuit breaker: fail fast after N failures for a cooldown window.
type Breaker struct {
	mu        sync.Mutex
	failures  int
	openedAt  time.Time
	threshold int
	cooldown  time.Duration
}

func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failures < b.threshold {
		return true
	}
	return time.Since(b.openedAt) > b.cooldown
}

func (b *Breaker) Report(err error) {
	if err == nil {
		b.mu.Lock()
		b.failures = 0
		b.mu.Unlock()
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	if b.failures == b.threshold {
		b.openedAt = time.Now()
	}
}
```

## Backpressure & Load Shedding

- Put bounded queues between producers/consumers.
- Track queue depth and time-in-queue; reject early when overloaded.
- Prefer 503 + `Retry-After` over “slow death” timeouts that pile up.

Bad (unbounded goroutines on load spikes):

```go
// ❌ BAD: spawns unbounded goroutines; melts under burst
for _, req := range requests {
	go handle(req)
}
```

Good (bounded queue + fixed workers):

```go
// ✅ GOOD: bounded queue provides backpressure
jobs := make(chan Job, 1000)
for i := 0; i < 32; i++ {
	go func() {
		for j := range jobs {
			_ = handle(j)
		}
	}()
}
```

## Graceful Shutdown

- Stop accepting new work.
- Cancel contexts, close listeners, drain in-flight requests, waitgroups.
- Ensure background goroutines have an owner and a stop path (no leaks).

## Idempotency & Exactly-once illusions

- Assume “at-least-once” delivery (queues, retries, client timeouts).
- Use unique constraints + upserts, outbox patterns, and de-dup keys.
