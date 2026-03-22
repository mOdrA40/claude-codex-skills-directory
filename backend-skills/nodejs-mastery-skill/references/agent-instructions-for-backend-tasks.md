# Agent Instructions for Node.js Backend Tasks

## Purpose

This guide helps an AI coding agent make safer backend decisions in Node.js services. It is not a language tutorial. It is an execution policy for common backend tasks.

## When Adding an Endpoint

Always check:

- request validation at the edge
- timeout posture for every outbound dependency
- stable error mapping
- request ID / trace correlation
- idempotency for side-effecting operations
- body size and rate-limit posture for public endpoints

Do not put business logic directly inside Express, Fastify, or Nest transport handlers.

## When Changing Database Logic

Always check:

- transaction boundaries
- retry safety
- unique constraints or idempotency keys
- mixed-version deploy compatibility
- whether slow queries or locks could turn a code change into an incident

Never assume schema and code roll out atomically.

## When Changing Queues or Background Jobs

Always check:

- worker ownership and shutdown behavior
- retry classification: transient vs terminal
- duplicate processing safety
- lag, age, and dead-letter visibility
- whether request-path work should actually move async

## When Debugging Production Issues

First classify the problem:

- dependency slowness
- event loop saturation
- queue backlog
- deployment regression
- unbounded retries or fan-out

Do not start by rewriting code. Start by identifying the bottleneck and blast radius.

## Non-Negotiables

- every outbound call needs a timeout
- every background task needs an owner
- every untrusted payload needs validation
- every deploy-sensitive change needs compatibility thinking
- every incident-prone path needs observability
