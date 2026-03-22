# Agent Instructions for Zig Backend Tasks

## Purpose

This guide helps an AI coding agent make safe backend decisions in Zig services where ownership, allocation, and overload posture must stay explicit.

## When Adding an Endpoint

Always check:

- request size and parsing limits
- allocator ownership for returned data
- timeout posture for outbound dependencies
- stable error mapping
- observability for incident debugging

## When Changing Memory or Concurrency Behavior

Always check:

- who allocates and who frees
- what happens on failure paths under `errdefer`
- who owns every thread or worker
- what happens under overload and queue growth
- whether one tenant can dominate memory or concurrency

## When Changing Data or Rollouts

Always check:

- compatibility across mixed-version deploys
- event and schema safety
- replay posture for workers
- whether rollback keeps the service operable

## Non-Negotiables

- no hidden allocator ownership
- no unbounded parsing or buffering
- no detached workers without stop path
- no rollout-sensitive change without compatibility plan
