# Agent Instructions for Go Backend Tasks

## Purpose

This guide helps an AI coding agent make production-safe decisions in Go services.

## When Adding an Endpoint or RPC

Always check:

- context propagation and cancellation
- explicit timeouts
- error taxonomy and mapping
- request size limits
- observability hooks
- idempotency where retries may occur

Prefer thin transport handlers and explicit dependency boundaries.

## When Changing Concurrency

Always check:

- ownership of goroutines
- shutdown behavior
- bounded fan-out
- queue depth and worker lag visibility
- whether retries can amplify load

## When Changing Data Paths

Always check:

- transaction boundaries
- lock or migration impact
- dual-version compatibility during rollout
- whether old and new binaries can coexist

## Non-Negotiables

- no unbounded goroutine spawning
- no outbound IO without timeout
- no log-and-return double logging
- no rollout-sensitive schema change without rollback plan
