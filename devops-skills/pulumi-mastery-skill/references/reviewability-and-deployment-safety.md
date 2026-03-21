# Reviewability and Deployment Safety

## Rules

- Deployment previews should be understandable by humans and highlight destructive risk clearly.
- Code review must focus on infrastructure semantics, not just language style.
- High-blast-radius changes deserve narrower scopes and stronger review.
- Rollback posture should be explicit for risky updates.

## Good Practices

- Keep code changes scoped to coherent infrastructure outcomes.
- Highlight replacements, security changes, network changes, and data-risk updates.
- Make dependency and ordering implications visible to reviewers.
- Use automation that improves trust rather than hiding preview meaning.

## Principal Review Lens

- Can a reviewer explain what will change in production?
- Which change is most dangerous even if it looks syntactically small?
- Are we batching unrelated risk into one deployment?
- What one improvement most increases deployment trust?
