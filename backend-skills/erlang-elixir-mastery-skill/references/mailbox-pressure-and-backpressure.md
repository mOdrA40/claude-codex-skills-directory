# Mailbox Pressure and Backpressure on the BEAM

## Purpose

Many BEAM incidents are not caused by node failure. They are caused by one or more processes receiving work faster than they can drain it. Mailbox growth is often the earliest visible sign.

## Core Principle

Mailbox growth is a systems symptom, not merely a slow-process symptom.

Possible root causes include:

- too much fan-in to one process
- slow downstream dependency calls inside handlers
- expensive serialization or DB work in the request path
- unbounded producer throughput
- retries that amplify queue pressure

## Why It Matters

A mailbox problem can cascade into:

- latency spikes
- scheduler imbalance
- memory growth
- stale work being processed too late to matter
- restart storms when overloaded processes fail repeatedly

## Design Rules

- do not centralize unrelated responsibilities in one GenServer
- keep messages small and purposeful
- isolate slow external IO away from hot coordination processes
- prefer bounded concurrency for consumers and fan-out work
- define what happens when backlog exceeds useful limits

## Bad vs Good

```text
❌ BAD
One GenServer receives every customer event, calls external APIs inline, and also updates cached counters.

✅ GOOD
Split coordination, external IO, and aggregation into separate supervised components with bounded work intake.
```

## Backpressure Options

Pick intentionally rather than letting the system decide under stress:

- reject new work
- queue with a hard cap
- shed low-priority work
- degrade non-critical enrichment
- slow producers explicitly

Different workloads deserve different failure semantics.

## Operational Signals

Watch:

- mailbox length for critical processes
- message processing latency
- queue age or consumer lag
- scheduler utilization
- restart intensity
- dependency latency correlated with mailbox growth

A healthy system does not just process messages eventually. It processes them while they still matter.

## Incident Triage Questions

- which process families show growing mailboxes?
- is the bottleneck compute, dependency latency, or message volume?
- are retries or requeues amplifying the problem?
- should old work be dropped, replayed later, or continue draining?
- what blast radius does this create for unrelated traffic?

## Review Checklist

- Does every hot process have bounded fan-in expectations?
- Can one tenant or one event type monopolize a shared process?
- Are slow calls isolated from high-volume coordination loops?
- Is backlog visibility good enough for on-call to act early?
- Is overload behavior defined before mailboxes become the implicit queue?
