# Agent Instructions for Erlang and Elixir Backend Tasks

## Purpose

This guide helps an AI coding agent make safe backend decisions on the BEAM, where process ownership, supervision, and mailbox health matter more than generic framework habits.

## When Adding Request or Consumer Logic

Always check:

- whether the responsibility needs a long-lived process at all
- message contracts and timeout semantics
- mailbox and concurrency posture
- dependency timeout and degradation behavior
- telemetry for operator visibility

## When Changing OTP Structure

Always check:

- blast radius of supervision changes
- what restarts together and why
- whether one process becomes a shared bottleneck
- whether retries or fan-out create queue pressure

## When Changing Data or Rollout Behavior

Always check:

- mixed-version compatibility
- worker and consumer replay safety
- whether old and new nodes can coexist during deploy
- whether rollback preserves correctness

## Non-Negotiables

- no catch-all GenServer owning unrelated responsibilities
- no unbounded async fan-out
- no hidden mailbox growth risk
- no deploy-sensitive change without compatibility posture
