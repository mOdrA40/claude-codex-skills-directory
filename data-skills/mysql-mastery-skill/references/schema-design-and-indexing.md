# Schema Design and Indexing (MySQL)

## Rules

- Schema design should serve transactional correctness and operational simplicity.
- Indexes exist to support actual filters, joins, uniqueness, and ordering needs.
- Every extra index adds write and storage cost.
- Data types and nullability choices affect correctness, storage, and application assumptions.

## Design Guidance

- Prefer clear relational modeling over premature denormalization.
- Review composite index order against real query predicates.
- Avoid using indexes to bandage fundamentally poor query shape forever.
- Make uniqueness and referential expectations explicit.

## Principal Review Lens

- Which high-value query is this index paying for?
- Are we designing schema around business invariants or around ORM convenience?
- What index will age poorly as data distribution changes?
- Which constraint is missing that application code is currently pretending to enforce?
