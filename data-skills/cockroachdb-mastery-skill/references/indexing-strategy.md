# Indexing Strategy (CockroachDB)

## Rules

- Indexes must support access paths and locality-sensitive reads.
- Every extra index has write and storage cost across replicas.
- Prefer indexes that reduce contention and full scans on hot tables.
- Review partial/secondary indexes with actual workload evidence.

## Principal Review Lens

- Which query is paying for this index?
- Does this index reduce cross-region work?
- Are we adding indexes instead of fixing poor key design?
