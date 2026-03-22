# Dependency Failure Decision Tree for Bun Services

## Database or Redis Slow

- reduce optional work
- protect pools and timeouts first
- avoid retries that outlive request usefulness
- degrade expensive reads when justified

## Webhook Provider or Third-Party API Down

- preserve idempotency state
- fail fast for non-critical paths if appropriate
- isolate retry policy from request handler
- avoid retry storms

## Queue Infrastructure Degraded

- protect request path from backlog propagation
- pause or slow toxic consumers
- preserve correctness before draining volume blindly

## Agent Heuristics

- classify dependency as critical, optional, or deferrable
- ask what gets degraded first
- ask what observable signal confirms containment works
