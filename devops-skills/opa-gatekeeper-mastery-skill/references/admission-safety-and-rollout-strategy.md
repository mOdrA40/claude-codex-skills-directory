# Admission Safety and Rollout Strategy

## Rules

- Admission control changes can become production outages instantly.
- Rollout strategy must be staged, observable, and reversible.
- Blocking policy should follow proof, not ambition.
- Validation of policy impact should happen before broad enforcement.

## Practical Guidance

- Start with audit or narrow-scope enforcement where uncertainty is high.
- Canary policy changes by namespace, cluster, or team when possible.
- Preserve strong observability around deny events and latency impact.
- Keep emergency disable or rollback workflows explicit.

## Principal Review Lens

- Which policy change is most likely to block critical production paths?
- Can the team safely stop enforcement during an emergency?
- What rollout step most reduces blast radius?
- Are we deploying policy with enough evidence?
