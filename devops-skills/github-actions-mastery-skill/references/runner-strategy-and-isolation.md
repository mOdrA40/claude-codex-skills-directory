# Runner Strategy and Isolation

## Rules

- Runner selection is a security, cost, and performance decision.
- Shared runners, self-hosted runners, and ephemeral runners have different blast-radius characteristics.
- Isolation should reflect secret usage, build trust level, and workload risk.
- Caching and artifact locality should not undermine security boundaries.

## Practical Guidance

- Separate privileged deployment workflows from general CI where needed.
- Keep runner lifecycle and patching owned explicitly.
- Watch queueing, starvation, and noisy-neighbor effects.
- Prefer ephemeral execution for high-risk or high-privilege contexts where practical.

## Principal Review Lens

- Which runner pool is the weakest trust boundary today?
- Are we optimizing cost at the expense of security or predictability?
- What workload most deserves stronger isolation?
- Can the team explain runner risk posture clearly?
