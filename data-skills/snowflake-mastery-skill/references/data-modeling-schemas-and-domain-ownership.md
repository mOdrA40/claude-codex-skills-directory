# Data Modeling, Schemas, and Domain Ownership

## Rules

- Data models should reflect domain ownership and consumer needs clearly.
- Shared schema chaos destroys trust and slows delivery.
- Governance must support discoverability and versioned evolution.
- Modeling for analytics still needs lifecycle discipline.

## Design Guidance

- Clarify ownership of curated, raw, and intermediate layers.
- Use naming and schema boundaries that reflect business domains.
- Make breaking change and deprecation workflows explicit.
- Align data product design with access and quality expectations.

## Principal Review Lens

- Which schema is most ownerless today?
- Are we building data products or just accumulating tables?
- What modeling choice most hurts discoverability and trust?
- Which domain boundary is currently too blurry?
