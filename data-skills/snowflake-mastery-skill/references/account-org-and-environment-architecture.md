# Account, Org, and Environment Architecture (Snowflake)

## Rules

- Account and environment design should reflect governance, compliance, and team boundaries.
- Separate experimentation, production, and regulated workloads intentionally.
- Organizational structure must support cost visibility and secure collaboration.
- Convenience-driven account sprawl creates governance debt.

## Design Guidance

- Clarify when to separate by account, region, org, or database boundary.
- Align platform boundaries with legal, residency, and support requirements.
- Keep ownership obvious for shared data assets.
- Document the rationale behind environment separation decisions.

## Principal Review Lens

- Which boundary is weakest for governance or cost accountability?
- Are we over-separating and hurting collaboration, or under-separating and creating risk?
- What architecture choice becomes hardest to undo later?
- Can operators explain why this environment model exists?
