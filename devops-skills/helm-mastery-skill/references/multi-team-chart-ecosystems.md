# Multi-Team Chart Ecosystems

## Rules

- Shared Helm ecosystems need ownership, versioning, and deprecation policy.
- Platform teams should publish clear chart contracts and upgrade guidance.
- Backward compatibility policy must be explicit for widely used charts.
- Ownerless or duplicated charts are operational debt.

## Ecosystem Guidance

- Track chart consumers and critical dependency chains.
- Distinguish platform-certified charts from experimental ones.
- Publish migration notes for breaking changes.
- Avoid fragmentation where every team forks the same chart differently.

## Principal Review Lens

- Which shared chart has the highest blast radius today?
- Are chart consumers trapped by unclear ownership or upgrade paths?
- What chart should be retired, split, or platform-standardized next?
- Is the ecosystem improving consistency or breeding snowflakes?
