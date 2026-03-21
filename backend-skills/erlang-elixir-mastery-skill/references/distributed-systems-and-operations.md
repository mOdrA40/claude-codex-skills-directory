# Distributed Erlang / Elixir Operations

## Purpose

BEAM systems are excellent for building resilient distributed services, but they do not remove distributed systems reality. Network partitions, mailbox buildup, overloaded dependencies, and bad cluster assumptions still break production systems.

## Cluster Thinking

Before enabling clustering, define:

- node discovery model
- trust boundaries
- cookie rotation and secret management
- what requires cluster-wide coordination
- how the system behaves during partitions
- whether the service can continue in degraded single-node mode

## Netsplits

A network partition is not a rare edge case. It is a design event.

Ask these questions:

- Which processes assume cluster visibility?
- Which jobs may double-run after a split?
- Which writes must remain single-owner?
- What is the reconciliation story after heal?

## Bad vs Good: Accidental Global Uniqueness

```text
❌ BAD
A periodic job assumes only one node will run it because "the cluster is usually stable".

✅ GOOD
Leadership or lease ownership is explicit, observable, and recoverable.
```

## Mailbox and Scheduler Health

High-concurrency systems fail quietly before they fail loudly.

Watch:

- mailbox length
- reductions
- process count
- scheduler utilization
- queue latency
- dependency timeout rate

## External Dependency Policy

For databases, queues, and HTTP dependencies:

- set explicit client timeouts
- define pool sizing intentionally
- classify retryable failures
- bound downstream fan-out
- expose telemetry for latency and saturation

## Release and Runtime Operations

A production release should define:

- startup checks
- readiness checks
- shutdown drain behavior
- migration policy
- remote shell access policy
- admin interface exposure policy

## Principal Incident Questions

- Is the problem local to one node or systemic?
- Are restarts helping or amplifying load?
- Which processes or mailboxes are growing abnormally?
- Are duplicate jobs or duplicate effects possible right now?
- What is the fastest safe blast-radius reduction?

## Minimum Operational Checklist

- Cluster membership is observable.
- Leadership-sensitive jobs are explicit.
- Timeouts and pool limits exist for dependencies.
- Telemetry answers mailbox, scheduler, and dependency health questions.
- Runbooks exist for netsplit, rollout, and dependency degradation.
