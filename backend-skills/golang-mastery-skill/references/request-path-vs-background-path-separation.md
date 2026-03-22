# Request Path vs Background Path Separation in Go Services

## Core Principle

Keep correctness-critical work on the request path. Move durable, slow, or fan-out-heavy work async when semantics and operability support it.

## Good Async Candidates

- notifications and webhooks
- optional enrichment
- imports and batch sync
- slow downstream publication

## Agent Questions

- what must complete before success is returned?
- does async movement require idempotency, queues, and replay safety?
- does this reduce or increase operational burden?
