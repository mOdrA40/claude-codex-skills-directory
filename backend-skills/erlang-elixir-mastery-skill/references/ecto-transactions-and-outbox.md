# Ecto Transactions and Outbox Patterns

## Purpose

Elixir services often combine database writes and async side effects. Without a transaction and outbox strategy, retries and crashes create inconsistent systems.

## Rules

- keep transaction boundaries explicit
- do not perform unreliable external side effects inside DB transactions casually
- record durable intent before async delivery when reliability matters
- define replay and deduplication semantics for consumers

## Bad vs Good

```text
❌ BAD
Insert DB row, call external webhook, then hope both succeed together.

✅ GOOD
Write domain change and outbox record transactionally, then deliver asynchronously with retry discipline.
```

## Review Questions

- what happens if the process crashes after commit but before publish?
- what happens if publish succeeds twice?
- how are outbox retries observed and controlled?
