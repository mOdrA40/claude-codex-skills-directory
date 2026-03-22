# Dependency Failure Decision Tree for Node.js Services

## Database Slow or Timing Out

- reduce expensive traffic
- protect pool exhaustion first
- avoid retries on non-idempotent writes
- consider degrading optional reads or enrichment

## Cache Unavailable

- decide fail-open vs fail-closed explicitly
- protect origin from stampede
- shorten expensive fallback paths if needed

## Third-Party API Failing

- classify dependency as critical or optional
- degrade optional features first
- avoid retry storms without budgets
- expose clear operator-facing signals

## Queue Dependency Degraded

- protect request path from backlog amplification
- pause toxic consumers if replay worsens pressure
- preserve correctness before throughput optics
