# API Evolution and Compatibility in Go Services

## Principle

Production APIs are long-lived contracts. Breaking compatibility casually is one of the fastest ways to create hidden operational failures across clients and internal services.

## Rules

- prefer additive change first
- deprecate before removing
- distinguish transport compatibility from business semantic compatibility
- version only when compatibility cannot be preserved safely
- coordinate event schema evolution as carefully as HTTP contracts

## Bad vs Good

```text
❌ BAD
A field meaning changes without changing validation, docs, or downstream expectations.

✅ GOOD
Compatibility rules are explicit, additive migration paths exist, and old/new versions can coexist during rollout.
```

## Review Questions

- can old and new clients work during rollout?
- what events or jobs still serialize the old shape?
- is semantic change masquerading as a non-breaking transport change?
