# Node.js Backend Architecture

## Core Position

A production Node.js backend should separate protocol concerns, orchestration, domain rules, and infrastructure adapters.

Preferred direction:

`transport -> service/use-case -> domain -> ports -> adapters`

## Why This Matters in Node.js

Node.js makes it easy to blur everything into route handlers because async IO is convenient. That convenience becomes operational debt when retries, transactions, and outbound calls are scattered across handlers.

## Bad vs Good: Route-Driven Architecture

```text
❌ BAD
Routes validate input, talk to DB, call webhooks, emit events, and map errors inline.

✅ GOOD
Routes are thin. A use-case coordinates policy, repositories, and side effects.
```

## Layout

```text
src/
├── index.ts
├── app.ts
├── config/
├── transport/
├── services/
├── domain/
├── ports/
├── adapters/
└── observability/
```

## Review Questions

- Where are retries owned?
- Where are errors translated to HTTP semantics?
- Which layer owns transactions?
- Can the domain be tested without HTTP or a database?
- Are side effects idempotent or replay-safe?
