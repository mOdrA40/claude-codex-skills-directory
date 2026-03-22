# SLO, Error Budget, and Release Gates for Bun Services

## Purpose

At scale, the question is not whether a Bun service can be fast. The question is whether the team can make disciplined release decisions when latency, failures, or dependency instability threaten user outcomes.

## Core Principle

SLOs and error budgets should influence release behavior.

If they do not change rollout policy, they are only reporting.

## Define Service Objectives

At minimum, define:

- critical endpoints or user journeys
- latency objectives for read and write paths
- availability objectives
- tolerated degradation modes
- which dependency failures count against the service promise

## Release Gates

Use stronger release posture when:

- error budget burn accelerates
- p99 latency regresses on critical endpoints
- dependency failure rate rises sharply
- queue lag or backlog exceeds safety threshold
- rollback time is uncertain

Possible actions:

- reduce canary size
- pause deploys
- require manual approval for risky changes
- disable expensive new features
- shift work from feature delivery to reliability fixes

## Review Questions

- what metrics can block a release automatically or operationally?
- which regressions matter enough to stop rollout?
- how quickly can rollback reduce user impact?
- does the team know which SLOs are business-critical versus nice-to-have?
- are release decisions tied to user outcomes or internal comfort metrics?

## Principal Heuristics

- Prefer simple release gates tied to real pain over elaborate dashboards nobody trusts.
- Error budget policy should constrain change when the system is unstable.
- If every release policy exception becomes normal, the gate design is weak.
