# Index Design (PostgreSQL)

## Rules

- Design indexes for actual predicates, joins, and ordering.
- Composite index order must match workload reality.
- Every index has write amplification and storage cost.
- Partial and covering indexes are powerful when used intentionally.

## Principal Review Lens

- Which hot query is this index paying for?
- Can one broader index replace multiple weak ones?
- Are we indexing around a bad query shape instead of fixing it?
