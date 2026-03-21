# Traffic Management and Rollout Control

## Rules

- Traffic policy should be readable and intentional.
- Retries, timeouts, circuit breakers, and canaries are business-affecting controls, not mesh decorations.
- Routing complexity should not outgrow the team's ability to debug it.
- Rollout safety depends on both policy correctness and observability.

## Failure Modes

- Retry storms amplifying dependency overload.
- Canary rules hiding broken subset or label assumptions.
- Timeout mismatches across app, mesh, and ingress.
- One virtual service becoming a platform-wide surprise.

## Principal Review Lens

- Which traffic rule is hardest to explain under pressure?
- Are we using the mesh to compensate for poor service behavior?
- What policy most increases blast radius if misconfigured?
- Can the team validate rollout behavior before production impact?
