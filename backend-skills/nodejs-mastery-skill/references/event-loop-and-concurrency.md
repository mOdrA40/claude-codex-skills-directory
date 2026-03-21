# Event Loop, Concurrency, and Cancellation

## Principle

Node.js concurrency is easy to misuse because non-blocking IO hides pressure until latency spikes. Principal-level design means understanding where work blocks the event loop, where fan-out multiplies load, and who owns cancellation.

## Guardrails

- Do not put CPU-heavy work on request paths without worker strategy.
- Bound concurrency for fan-out operations.
- Use cancellation for outbound dependencies.
- Separate request traffic from background work.

## Bad vs Good: Unbounded Fan-Out

```typescript
// ❌ BAD
await Promise.all(items.map(processItem))

// ✅ GOOD
await pool.run(items, { concurrency: 8 }, processItem)
```

## Cancellation

For outbound fetch or SDK calls, define deadlines and propagate them consistently.

## Event Loop Delay

For latency-sensitive services, monitor event-loop lag. High event-loop delay can mean:

- synchronous CPU work
- huge JSON serialization
- excessive compression
- overloaded logging
- crypto or parsing work on the main thread
