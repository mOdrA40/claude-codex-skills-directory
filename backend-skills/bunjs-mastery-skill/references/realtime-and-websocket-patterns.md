# Realtime and WebSocket Patterns in Bun Services

## Purpose

Realtime systems fail through connection lifecycle mistakes, missing backpressure, poor auth refresh strategy, and unbounded fan-out.

## Rules

- define connection ownership and lifetime
- authenticate before subscription
- track subscription scope explicitly
- bound fan-out and queueing per connection
- define disconnect and resume behavior

## Bad vs Good

```text
❌ BAD
One connection can subscribe to arbitrary high-volume channels with no quota.

✅ GOOD
Subscriptions are authorized, bounded, and observable.
```

## Operational Questions

- what is the per-connection memory cost?
- how is slow consumer behavior handled?
- can one tenant overload broadcast work?
- what happens on deploy or node restart?
