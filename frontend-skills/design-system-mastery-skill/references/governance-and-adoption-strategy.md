# Governance and Adoption Strategy

## Principle

A design system fails when it is either too weak to guide teams or too rigid to earn adoption. Governance must balance consistency and pragmatism.

## Governance Questions

- who approves new primitives
- who owns breaking changes
- how do product teams request exceptions
- how is adoption measured

## Adoption Heuristics

### Make the system easier than bypassing it

If consuming teams face more friction using the system than avoiding it, governance will eventually lose.

### Distinguish temporary exceptions from permanent forks

Exceptions are inevitable. The problem begins when exceptions are undocumented, unaudited, and never retired.

### Measure meaningful adoption

Adoption is not just package install count. It includes:

- real component usage
- reduction in duplicated UI patterns
- migration of high-risk surfaces
- fewer local accessibility regressions

## Operating Model

Good governance usually distinguishes between:

- platform-owned primitives
- product-owned compositions
- temporary exceptions
- deprecated legacy surfaces

Without those categories, every adoption conversation becomes a negotiation from scratch.

## Exception Lifecycle

An exception should answer:

- why the system path is insufficient
- who approved the exception
- how long it is expected to live
- what migration path would remove it later

## Failure Modes

- system as gatekeeper only
- uncontrolled local forks
- no clear migration path for old components

### Governance theater

The process looks formal, but teams still solve urgent product work with ad hoc local components because the system response time is too slow.

### Permanent temporary exceptions

Teams create escape hatches for short-term product pressure, but nobody owns retiring them, so they become the real system over time.

## Review Questions

- what category of exceptions is growing fastest and why?
- where is governance creating delay without improving quality?
- what part of the design system lacks a clear ownership model today?
