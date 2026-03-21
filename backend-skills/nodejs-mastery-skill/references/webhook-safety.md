# Webhook Safety and Delivery Discipline

## Principle

Webhook systems are distributed systems with untrusted boundaries. They fail through retries, signature mistakes, replay bugs, and poor observability.

## Rules

- verify signatures before trust
- set timeout and retry policy intentionally
- make delivery idempotent
- define replay tolerance
- distinguish provider downtime from bad payloads
- log delivery attempts without leaking secrets

## Review Questions

- can a webhook be replayed safely?
- how are duplicate deliveries handled?
- what happens when the provider slows down or sends malformed payloads?
