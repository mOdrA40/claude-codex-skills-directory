# Database Boundaries and Transaction Design

## Purpose

Many backend defects are not code bugs but transaction-boundary mistakes. Node.js services should define where invariants live, what is transactional, and what is eventually consistent.

## Rules

- keep transaction scope explicit and small
- do not mix local transaction state with unreliable remote side effects casually
- define conflict and retry behavior for concurrent writes
- know which reads can be stale and which cannot

## Bad vs Good

```text
❌ BAD
A request handler opens a transaction, performs multiple remote calls, then tries to commit.

✅ GOOD
Transactions protect local invariants; remote side effects are coordinated via durable asynchronous patterns when needed.
```

## Review Questions

- what invariant must be atomic?
- what can be eventually consistent safely?
- what happens when a commit succeeds but remote work fails?
