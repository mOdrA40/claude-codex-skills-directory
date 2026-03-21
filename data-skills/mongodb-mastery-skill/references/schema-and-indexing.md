# Schema and Indexing (MongoDB)

## Modeling Rules

- Model for query shape first.
- Embed when data is read together and growth is bounded.
- Reference when cardinality or update independence matters.
- Keep schema versioning intentional when documents evolve.

## Index Rules

- Every hot query should have an index story.
- Compound index order must match real predicates and sorting.
- Avoid index sprawl that punishes writes without clear read benefit.

## Principal Review Lens

- Will document growth cause relocation or update pain?
- Which indexes are actually paying for themselves?
- Is the shard key aligned with hottest access paths?
