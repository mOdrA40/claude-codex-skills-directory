# Webhooks, Idempotency, and Retries in Bun Services

## Purpose

Webhook handling looks simple until retries, duplicate delivery, downstream slowness, and signature validation turn it into an incident source. This guide covers a production-safe baseline for Bun services receiving or sending webhooks.

## Core Principle

Treat every webhook as an at-least-once delivery mechanism.

That means your design must assume:

- duplicate delivery
- delayed delivery
- out-of-order delivery
- dependency slowness during processing
- replay attempts from legitimate providers or attackers

## Inbound Webhook Rules

- verify authenticity before expensive processing
- bound body size and parse time
- persist or record idempotency before side effects
- acknowledge quickly when async processing is the safer path
- separate transport acknowledgement from business workflow completion

## Signature Verification

Do not normalize, reserialize, or mutate the body before verification when the provider expects raw payload verification.

```typescript
// ❌ BAD: parsing first may change the canonical payload.
const body = await c.req.json()
verifySignature(JSON.stringify(body), signature)
```

```typescript
// ✅ GOOD: verify against the raw body first.
const rawBody = await c.req.text()
verifySignature(rawBody, signature)
const payload = JSON.parse(rawBody)
```

## Idempotency Strategy

Use a stable key such as:

- provider event id
- delivery id
- external object id plus event type
- a domain command key derived from the external event

Store idempotency state before or together with side effects when possible.

### Minimum states

- received
- processing
- completed
- failed_terminal

This makes retries observable and prevents ambiguous reprocessing.

## Bad vs Good: Retry Safety

```text
❌ BAD
Receive webhook -> call domain side effect -> crash before recording completion.
Retry causes duplicate external effect.

✅ GOOD
Receive webhook -> reserve idempotency key -> execute side effect -> mark completion.
```

## Outbound Webhook Sending

When your Bun service publishes webhooks:

- use explicit timeout budgets
- classify retryable vs terminal failures
- sign payloads consistently
- keep delivery attempts observable
- use dead-letter or operator review for long-failing deliveries

Do not retry forever in-process. Delivery policy belongs to a queue or explicit delivery component.

## Observability Minimum

Track at least:

- webhook receive rate
- verification failures
- duplicate deliveries
- processing latency
- retry count
- dead-letter count
- downstream dependency latency involved in processing

Logs should include stable fields like `provider`, `eventType`, `eventId`, `deliveryId`, and `tenantId` when applicable.

## Review Questions

- What prevents duplicate side effects?
- Can slow downstream dependencies block acknowledgement too long?
- Is signature verification happening on the correct raw input?
- How are poison events isolated from healthy traffic?
- Can operators replay safely without manual database surgery?

## Principal Heuristics

- Prefer fast acknowledgement plus durable async processing for expensive handlers.
- Treat idempotency storage as part of the business boundary, not optional infrastructure.
- If replay is required, make replay safe by design instead of relying on operator caution.
