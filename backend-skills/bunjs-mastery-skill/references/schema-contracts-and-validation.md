# Schema Contracts and Validation in Bun APIs

## Principle

Validation is only valuable if it protects the right boundaries consistently and if the system treats contract drift as an operational risk.

## Rules

- validate inbound request shape intentionally
- validate important downstream responses when trust or impact requires it
- keep schema ownership explicit
- align runtime validation and TypeScript types instead of letting them drift

## Review Questions

- which contracts are allowed to be best-effort and which are not?
- are runtime validators and TS types diverging?
- what happens when a downstream response changes shape silently?
