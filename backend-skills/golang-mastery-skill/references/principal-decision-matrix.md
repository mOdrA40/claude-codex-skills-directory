# Principal Decision Matrix for Go Services

## Purpose

Go services often start simple and stay alive for years. This guide helps choose appropriate complexity before the system becomes harder to evolve safely.

## Decision Axes

- request criticality
- dependency count
- concurrency complexity
- data correctness requirements
- public contract stability
- operator burden

## When to Keep It Small

Keep a service simple when:

- one main API surface exists
- few dependencies exist
- business rules are modest
- operator burden is low

## When to Introduce Stronger Boundaries

Increase structure when:

- request and worker paths diverge
- retries and idempotency matter
- multiple teams contribute
- incident response needs clearer ownership

## When Not to Split Services

Do not split because:

- the org likes microservices
- folders look crowded
- different developers want separate repos

Split only when ownership, scaling, and blast radius justify the cost.

## Review Questions

- what complexity are we adding and what risk does it remove?
- what incident becomes easier with this design?
- what coordination cost are we creating?
