# Reviewability and Plan Safety

## Rules

- Plans should be understandable by reviewers who did not write the change.
- A safe plan review focuses on semantics, dependencies, and replacement risk.
- Reduce noise so reviewers can see destructive or surprising changes.
- Human review is not optional theater for high-blast-radius infrastructure.

## Good Practices

- Separate formatting churn from infrastructure meaning.
- Highlight replacements, IAM changes, networking changes, and data-risk changes.
- Keep module and provider upgrades scoped when possible.
- Make rollback posture explicit for risky applies.

## Principal Review Lens

- Can a reviewer tell what user-visible behavior might change?
- Which resource replacement is easy to miss but expensive to recover from?
- Are we batching unrelated infrastructure changes together?
- What would make this plan easier to trust under time pressure?
