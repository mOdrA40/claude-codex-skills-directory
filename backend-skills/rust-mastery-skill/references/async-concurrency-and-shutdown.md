# Async Concurrency and Shutdown in Rust Services

## Purpose

Async Rust often looks correct before it is operationally correct. A principal-level service must define ownership for tasks, cancellation, bounded concurrency, and shutdown behavior explicitly.

## Rules

- every spawned task has an owner
- every task has a shutdown path
- fan-out concurrency is bounded
- timeouts are explicit for IO and pool acquisition
- shared mutable state is justified, not defaulted to

## Bad vs Good

```rust
// ❌ BAD: detached task with no shutdown path.
tokio::spawn(async move {
    run_consumer_loop().await;
});
```

```rust
// ✅ GOOD: task respects cancellation.
tokio::spawn(async move {
    tokio::select! {
        _ = shutdown.cancelled() => {}
        result = run_consumer_loop() => {
            if let Err(error) = result {
                tracing::error!(%error, "consumer loop failed");
            }
        }
    }
});
```

## Shutdown Sequence

1. fail readiness
2. stop accepting new traffic
3. cancel workers and background tasks
4. drain in-flight work with deadlines
5. close pools and clients
6. exit cleanly

## Review Questions

- Can this task outlive the service accidentally?
- What happens under dependency slowness?
- Is shutdown correctness tested?
