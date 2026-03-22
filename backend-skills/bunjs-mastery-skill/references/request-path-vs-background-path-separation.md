# Request Path vs Background Path Separation in Bun Services

## Core Principle

Keep the request path focused on correctness-critical work. Move durable, slow, or fan-out-heavy work async when semantics allow.

## Good Async Candidates

- webhooks and notifications
- optional enrichment
- batch imports
- slow third-party publication

## Agent Questions

- what must complete before acknowledging success?
- does async handling require idempotency and replay safety?
- will moving work async reduce or increase operational complexity?
