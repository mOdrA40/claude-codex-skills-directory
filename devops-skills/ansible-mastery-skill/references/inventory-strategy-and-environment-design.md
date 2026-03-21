# Inventory Strategy and Environment Design

## Rules

- Inventory should model real infrastructure boundaries, environments, and ownership.
- Dynamic inventory is powerful but requires discipline and observability.
- Group hierarchy should improve readability, not hide exceptions and drift.
- Environment differences should be explicit and reviewable.

## Common Mistakes

- Treating inventory as a historical junk drawer.
- Relying on variable precedence that few humans can reason about.
- Mixing immutable and mutable infrastructure assumptions carelessly.
- Letting emergency changes create long-lived inventory lies.

## Principal Review Lens

- Can a new operator infer scope and ownership from inventory alone?
- Which group or variable pattern is most dangerous under pressure?
- Are environment boundaries real or just naming conventions?
- What inventory cleanup would reduce confusion most?
