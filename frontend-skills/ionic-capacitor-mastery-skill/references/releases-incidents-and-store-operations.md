# Releases, Incidents, and Store Operations

## Principle

Hybrid app releases combine web-style iteration pressure with mobile-store constraints. Teams must know which problems can be fixed quickly and which require full binary release discipline.

## Common Failure Modes

- underestimating plugin or native dependency impact on release risk
- poor visibility into version-specific incidents
- rollback assumptions that ignore store-review or user-update realities

## Review Questions

- what release path exists for urgent regressions?
- which incidents are tied to binary version vs server-side behavior?
- how is app version correlated to crash and support signals?
