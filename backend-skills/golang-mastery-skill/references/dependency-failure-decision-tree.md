# Dependency Failure Decision Tree for Go Services

## Database Slow or Pool Exhausted

- reduce optional read/write load
- protect critical transactions first
- avoid write retries without idempotency
- inspect lock and pool wait posture

## Cache Unavailable

- choose fail-open vs fail-closed deliberately
- protect origin from thundering herd
- shorten fallback path if needed

## Third-Party API Failing

- classify as critical or optional
- degrade optional features first
- do not multiply retries across layers

## Queue or Broker Degraded

- pause toxic consumers if replay amplifies pressure
- protect request path from backlog bleed
- preserve correctness before throughput optics
