# Pods, Deployments, and Rollouts

## Rules

- Health probes must reflect real readiness, not optimistic startup.
- Rollout strategy should match blast radius tolerance.
- Pod lifecycle and graceful termination must be explicit.
- Avoid rollout settings that amplify bad releases.

## Principal Review Lens

- What happens during a bad rollout under peak traffic?
- Do probes reflect actual dependency readiness?
- How long does graceful shutdown really take?
