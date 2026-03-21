# Event-Driven Design and Outbox Patterns in Node.js

## Principle

Publishing events is easy. Publishing events reliably while preserving transaction truth, replay safety, and observability is hard.

## Rules

- define whether the event is notification, integration fact, or workflow trigger
- do not treat event publication as a side effect you can casually `await` after a DB write when reliability matters
- prefer outbox or equivalent durability when state change and event publication must stay coherent
- version events intentionally and deprecate carefully

## Event Categories

### Domain event

Represents a meaningful business fact.

### Integration event

Designed for other services or partners.

### Internal workflow event

Used to continue asynchronous processing.

These should not all be treated the same operationally.

## Bad vs Good

```text
❌ BAD
Write DB row, publish event directly, and hope a crash does not happen between them.

✅ GOOD
Persist durable intent and deliver asynchronously with retries and visibility.
```

## Review Questions

- what guarantees exist between state change and event publication?
- who owns replay or redelivery?
- how are duplicate consumers kept safe?
