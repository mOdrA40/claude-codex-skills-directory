# Production Decision Matrix for Bun Services

## Purpose

This guide helps decide when Bun is the right operational choice and which patterns fit the service shape best.

## Decision Matrix

### Question 1: Is the service mostly IO-bound?

- **Yes**: Bun is often a strong fit.
- **No**: investigate CPU-heavy paths and runtime isolation strategy first.

### Question 2: Does framework portability matter?

- **Yes**: favor Web-standard patterns and Hono-style layering.
- **No**: Elysia or tighter runtime-specific conventions may be acceptable.

### Question 3: Are background jobs first-class in this service?

- **Yes**: define ownership, shutdown, retry, and metrics before optimizing runtime speed.
- **No**: keep the service focused and simple.

### Question 4: Are schema changes frequent and risky?

- **Yes**: strengthen migration and compatibility discipline.
- **No**: keep deployment process simple but still explicit.

## Architectural Recommendations

- small API + few deps -> thin layered service
- public API + multiple side effects -> stronger use-case boundaries
- realtime + subscriptions -> explicit connection and fan-out governance
- heavy async workflows -> dedicated workers and clear replay semantics

## Review Questions

- are we choosing Bun for real operational benefit or just trend value?
- which runtime assumption would hurt us most during incidents?
- what boundary is doing the most risk reduction?
