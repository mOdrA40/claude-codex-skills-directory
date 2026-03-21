# Storage Backends and Retention Tradeoffs

## Rules

- Backend choice affects search latency, retention, cost, and operator complexity.
- Retention should match debugging value and regulatory need.
- Storage reliability must be considered part of trace reliability.
- Trace promises should reflect actual backend durability and throughput.

## Design Guidance

- Match storage backend to query style and scale realities.
- Consider index/search cost, long-term retention, and operational skills.
- Test restore and access assumptions if history matters materially.
- Keep backend migration risk visible to platform owners.

## Principal Review Lens

- Which storage choice imposes the highest long-term tax?
- Are we over-retaining low-value traces?
- What backend failure most harms incident response?
- Can the team explain trace durability honestly?
