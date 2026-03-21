# Performance and Change Detection in Angular

## Principle

Performance work in Angular should focus on rendering scope, change-detection cost, route boundaries, and async churn rather than cargo-cult micro-optimizations.

## Investigation Model

Angular performance issues often come from one of these categories:

- route load and dependency cost
- broad state changes causing too much UI churn
- change-detection work that is wider than the user value justifies
- async flows that trigger too many updates or retries

Teams should classify the issue before optimizing anything.

## Common Failure Modes

- over-shared state causing broad UI updates
- routes with bundle or data dependency bloat
- teams using performance APIs mechanically without understanding real bottlenecks

### Change-detection superstition

The team changes detection strategy mechanically without proving the dominant user pain is actually caused by that layer.

### Route architecture masking performance cost

The page looks like a rendering problem, but the deeper issue is route ownership, async data shape, or state churn.

## Performance Heuristics

### Find the widest update boundary first

The biggest gain often comes from narrowing the scope of updates rather than micro-optimizing templates.

### Tie performance work to user-visible pain

The important question is not "what looks expensive" but "what actually slows the user journey we care about".

## Review Questions

- what user-visible pain are you trying to remove?
- is the main issue route load, change detection, or async churn?
- what boundary would reduce the most unnecessary work?
- what optimization would the team most likely try first even if it is the wrong one?
