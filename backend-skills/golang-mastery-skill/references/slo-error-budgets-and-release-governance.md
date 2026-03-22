# SLO, Error Budgets, and Release Governance in Go Services

## Purpose

Reliable Go services need more than clean code and benchmarks. They need clear service promises and release discipline that changes when reliability is under stress.

## Core Principle

SLOs should drive behavior.

If latency regressions, dependency instability, or repeated incidents do not change rollout posture, then the service has reporting, not governance.

## Define Objectives

At minimum, define:

- critical endpoints or workflows
- latency objectives by path class
- availability objectives
- tolerated degraded modes
- which downstream failures count against the promise

## Error Budget Use

When budget burn accelerates:

- reduce release risk
- pause low-value changes
- focus work on dominant failure classes
- tighten canary or rollout gates
- remove optional expensive traffic where justified

## Review Questions

- which user journeys actually define the service promise?
- what metric should halt rollout?
- how quickly can rollback recover the budget burn?
- are queue lag and async completion part of the service promise?
- does the team change priorities when the service is unstable?

## Principal Heuristics

- Prefer a few trusted SLOs over a dashboard zoo.
- Tie release policy to user outcomes, not engineer intuition alone.
- If budget policy is always waived, governance is weak.
