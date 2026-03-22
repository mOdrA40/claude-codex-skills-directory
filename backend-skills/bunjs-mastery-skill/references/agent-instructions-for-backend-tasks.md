# Agent Instructions for Bun Backend Tasks

## Purpose

This guide helps an AI coding agent make safe, production-aware decisions in Bun backend services.

## When Adding an Endpoint

Always check:

- request and config validation
- timeout and abort posture for all outbound calls
- stable error envelopes
- observability fields such as request ID and route
- file upload and webhook authenticity boundaries when relevant

Keep Hono/Elysia transport concerns out of service logic.

## When Changing Data or Migrations

Always check:

- compatibility during mixed-version deploys
- transaction safety
- idempotency for writes and retries
- whether Drizzle or SQL migrations can lock or surprise production traffic

## When Touching Background Work

Always check:

- ownership of worker lifecycle
- concurrency caps
- retry and dead-letter semantics
- whether shutdown drains safely
- whether this work belongs off the request path

## When Debugging

Classify first:

- runtime saturation
- dependency timeout storm
- queue backlog
- rollout regression
- tenant/noisy-neighbor issue

## Non-Negotiables

- no unbounded outbound IO
- no hidden global mutable state
- no retry strategy without idempotency thinking
- no deploy-sensitive schema change without compatibility posture
