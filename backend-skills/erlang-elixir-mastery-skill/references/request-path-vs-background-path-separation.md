# Request Path vs Background Path Separation on the BEAM

## Core Principle

Keep correctness-critical work on the request path. Move durable, slow, or fan-out-heavy work async when semantics and operability support it.

## Good Async Candidates

- notifications and fan-out publication
- optional enrichment
- imports or batch sync
- slow downstream delivery

## Agent Questions

- what must complete before success is returned?
- does async movement require idempotency, queues, and replay safety?
- will moving work async reduce or increase mailbox and backlog risk?
