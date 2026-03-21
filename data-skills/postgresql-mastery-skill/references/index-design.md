# Index Design (PostgreSQL)

## Rules

- Design indexes for actual predicates, joins, and ordering.
- Composite index order must match workload reality.
- Every index has write amplification and storage cost.
- Partial and covering indexes are powerful when used intentionally.

## Index Heuristics

### Every index should pay rent

An index is justified when it materially improves important workload behavior enough to offset write cost, storage growth, maintenance overhead, and planner complexity.

### Design for query families, not isolated SQL snippets

The best indexes often support a class of hot access patterns rather than one individually optimized query that may change tomorrow.

### Indexing should not hide model problems forever

Indexes can rescue performance, but they should not become a permanent substitute for fixing wasteful query choreography, weak pagination design, or poor table access patterns.

## Common Failure Modes

### Index pile-up

Teams keep adding indexes per incident or ticket until write cost and planner complexity grow without a coherent workload strategy.

### Composite order mismatch

The index looks plausible, but column order does not match the real filter and sort shape that matters in production.

### Query-shape denial

The real issue is that the application makes too many queries or uses unstable access patterns, but index additions are used as the only response.

## Principal Review Lens

- Which hot query is this index paying for?
- Can one broader index replace multiple weak ones?
- Are we indexing around a bad query shape instead of fixing it?
- Which index would we delete first if write cost became the dominant pain?
