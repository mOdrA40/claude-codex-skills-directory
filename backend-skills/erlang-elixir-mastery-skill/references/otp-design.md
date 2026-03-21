# OTP Design for Production Systems

## Purpose

OTP is not a collection of conveniences. It is the operating model of a BEAM system. Production Erlang and Elixir services should be designed around supervision boundaries, process ownership, restart semantics, and observable failure behavior.

## Core Principle

Do not ask, "Which process should run this code?"

Ask:

- Which responsibility deserves its own failure boundary?
- What state is local to that responsibility?
- What restart policy makes sense?
- What signals show that it is unhealthy?

## Supervision Tree Design

Design trees around blast radius.

Good supervision boundaries usually separate:

- request-serving infrastructure
- background jobs
- external connection pools
- cache refreshers
- streaming consumers
- dynamic per-tenant or per-session workers

## Restart Strategy Heuristics

### `:one_for_one`

Use when processes fail independently.

### `:rest_for_one`

Use when later children depend on earlier ones.

### `:one_for_all`

Use rarely, only when the children form one integrity unit.

If you cannot explain why several children must restart together, do not use broad restart coupling.

## Bad vs Good: Wrong Blast Radius

```text
❌ BAD
A single supervisor owns:
- Phoenix endpoint
- database repo
- webhook consumers
- cache warmer
- billing sync loop

One noisy background component can destabilize unrelated request traffic.

✅ GOOD
Separate supervision domains by operational concern.
- request path infrastructure
- async consumers
- background sync jobs
- dynamic workloads
```

## GenServer Discipline

Use a `GenServer` when you need:

- serialized access to state
- a process boundary with restart semantics
- timed work or message handling

Do not use a `GenServer` merely to make a module feel important.

## Message Contracts

Every process with public messages should define:

- supported message shapes
- expected reply semantics
- timeout behavior
- backpressure behavior
- whether callers may retry

## Process State Guidance

Keep state:

- minimal
- reconstructable when possible
- explicit about invariants
- free from cached garbage that will grow mailboxes or heap pressure

## Principal Review Questions

- What restarts together, and why?
- Which process is a throughput bottleneck?
- What happens if one dependency becomes slow for 10 minutes?
- Which processes are expected to be ephemeral vs long-lived?
- How are failures surfaced to operators?
- Is there a dead-letter or replay story for critical messages?
