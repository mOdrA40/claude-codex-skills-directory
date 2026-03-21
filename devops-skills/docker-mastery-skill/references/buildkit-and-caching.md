# BuildKit and Caching

## Rules

- Cache strategy should optimize developer and CI feedback loops.
- Use BuildKit features intentionally, not decoratively.
- Remote caching must balance speed and trust boundaries.
- Build performance matters when fleets or monorepos grow.

## Principal Review Lens

- Which step dominates build time?
- Is cache reuse safe across branches and environments?
- Are we paying complexity for negligible gain?
