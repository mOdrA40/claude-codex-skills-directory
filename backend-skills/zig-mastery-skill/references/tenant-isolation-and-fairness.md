# Tenant Isolation and Fairness in Zig Backends

## Purpose

At scale, multi-tenant failures are rarely pure authorization bugs. They are often fairness bugs: one tenant, job class, or workload shape consumes more memory, concurrency, queue space, or downstream capacity than intended.

## Core Principle

Isolation is not only about data separation. It is also about protecting latency, throughput, and resource availability across tenants.

## Design Rules

- propagate tenant identity across sync and async boundaries
- define per-tenant limits for expensive operations where needed
- isolate queue depth, concurrency, or rate limits when blast radius justifies it
- make noisy-neighbor detection observable
- protect shared dependencies from one tenant's pathological workload

## Isolation Layers

Useful controls may exist at several levels:

- request admission
- queue partitioning
- worker concurrency limits
- cache quotas
- downstream connection or token budgets
- per-tenant circuit breaking or temporary throttling

## Review Questions

- can one tenant exhaust memory through valid-but-expensive requests?
- can one tenant monopolize worker pools or queue age?
- where is tenant identity lost during retries or background processing?
- how would operators detect and mitigate a noisy neighbor quickly?
- what fairness policy is intentional versus accidental?

## Principal Heuristics

- Global limits without tenant-awareness often preserve the system while failing fairness.
- Perfect isolation is expensive; define where the blast radius truly matters.
- If a single tenant incident requires manual detective work during peak traffic, observability is insufficient.
