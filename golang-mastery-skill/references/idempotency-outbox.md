# Idempotency & Outbox (Money-moving / Exactly-once-ish)

## The truth

Distributed systems are at-least-once by default. Your code must survive:
- client retries (timeouts)
- duplicated messages
- partial failures (DB committed, publish failed)

## Idempotency keys (API)

For endpoints with side effects (payments, create order, send email):
- Accept `Idempotency-Key` header.
- Store the key + request fingerprint + result (or canonical resource ID) with a TTL.
- On duplicates: return the same result without repeating the side effect.

## Outbox pattern (DB transaction → publish)

Why:
- You can’t atomically commit DB + publish to a queue without a coordinator.

Pattern:
1) In a DB transaction:
   - write business data
   - write outbox row (event payload + type + key)
2) Commit.
3) A background worker reads unpublished outbox rows and publishes.
4) Mark as published (idempotently). Publisher must handle duplicates.

Minimum table shape:
- `id` (pk)
- `type`
- `key` (dedup key)
- `payload` (json)
- `created_at`
- `published_at` (nullable)
- `attempts`, `last_error` (optional)

## Good vs bad

Bad: publish inside transaction (can deadlock/timeout; can publish then rollback):

```go
tx := db.Begin()
defer tx.Rollback()
_ = publish(msg) // unsafe
_ = tx.Commit()
```

Good: outbox + async publish.

## De-dup (consumer side)

- Use `key` or `event_id` with a uniqueness constraint in the consumer’s DB.
- If the operation is naturally idempotent (upserts with unique constraints), lean on that.

