# Safe Rollout Patterns for High-Traffic Node.js Services

## Principles

- prefer canaries over big-bang deploys
- keep readiness traffic-safe, not merely boot-successful
- ensure old and new versions can coexist during rollout
- define rollback trigger before deploy starts

## Good Patterns

- canary by route or traffic slice
- feature flags for risky optional behavior
- stagger worker rollout separately from request-serving nodes
- halt rollout on p99, error rate, or queue lag regression

## Anti-Patterns

- deploy plus migration plus config flip at once
- treat autoscaling as rollout safety
- ignore queue consumers during compatibility changes

## Agent Heuristics

- Ask what signal should stop rollout.
- Ask how rollback reduces user pain quickly.
- Ask whether background work is compatible across versions.
