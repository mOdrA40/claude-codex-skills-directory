# Dependency Failure Decision Tree for Rust Services

## Database Slow or Pool Exhausted

- protect critical transactions first
- reduce optional read/write paths
- avoid write retries without idempotency
- inspect lock and acquisition latency before scaling blindly

## Cache or Derived Data Dependency Failing

- decide fail-open vs fail-closed explicitly
- protect origin from fallback stampede
- reduce expensive misses when necessary

## Third-Party API Failing

- classify critical vs optional dependency
- degrade optional behavior first
- avoid retry storms across async tasks and request path

## Broker or Queue Dependency Degraded

- pause toxic consumers if replay worsens pressure
- protect request path from backlog bleed
- prioritize correctness over headline throughput
