# Agent Instructions for Rust Backend Tasks

## Purpose

This guide helps an AI coding agent make safe backend decisions in Rust services without over-indexing on language cleverness.

## When Adding an Endpoint

Always check:

- boundary validation
- typed error mapping
- timeout and cancellation posture
- health/readiness and tracing implications
- idempotency for side-effecting operations

Keep transport separate from domain and avoid leaking framework details inward.

## When Adding Async or Background Work

Always check:

- task ownership
- cancellation path
- queue caps and backpressure
- retry classification
- observability for lag, age, and failure rate

## When Touching Data or Compatibility

Always check:

- mixed-version rollout safety
- event and schema compatibility
- transaction boundaries
- replay and duplicate safety
- whether rollback is operationally realistic

## Non-Negotiables

- no `unwrap()` on real production paths
- no detached tasks without owner and shutdown path
- no unsafe expansion without invariants
- no rollout-sensitive change without compatibility thinking
