# Bun Runtime Operability

## Goal

Fast runtimes still need production discipline. Bun services should define clear operational behavior around startup, shutdown, dependency errors, and observability.

## Minimum Baseline

- validated config at startup
- request ID propagation
- health and readiness endpoints
- explicit outbound timeouts
- graceful shutdown
- dependency saturation metrics
- clear crash signals and log schema

## Review Questions

- Is runtime-specific behavior observable during incidents?
- Are Web-standard APIs used consistently across code paths?
- Are Node compatibility assumptions sneaking into Bun-specific deployments?
