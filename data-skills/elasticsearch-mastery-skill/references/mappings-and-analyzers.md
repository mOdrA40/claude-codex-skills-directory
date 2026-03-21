# Mappings and Analyzers (Elasticsearch)

## Rules

- Important fields deserve explicit mappings.
- Analyzer choice is product behavior, not only text plumbing.
- Separate exact-match and full-text needs intentionally.
- Reindex cost should influence mapping ambition.

## Schema Heuristics

### Mapping choices are long-term commitments

Field type, analyzer, multi-field strategy, and dynamic behavior all influence storage, relevance, query performance, and future migration cost.

### Analyzer design should reflect user language

Tokenization, stemming, synonyms, and normalization should be reviewed as product behavior that affects what users believe search means.

### Dynamic convenience must stay governed

Dynamic mapping can accelerate adoption, but without guardrails it creates field sprawl, type inconsistency, and much more expensive recovery later.

## Common Failure Modes

### Mapping convenience debt

Early flexibility feels productive until reindex cost, field explosion, or search inconsistency turns it into a platform tax.

### Analyzer tweak treated as harmless

The team changes text behavior as if it were internal plumbing, even though the user-facing relevance effect may be significant.

### Exact vs text confusion

Fields are used for multiple incompatible semantics without a clear strategy, making queries and relevance harder to reason about.

## Principal Review Lens

- Which field is over-flexible and will hurt later?
- Do analyzers match user language and query behavior?
- What mapping change would force painful reindexing?
- Which mapping shortcut today is most likely to become a major migration or relevance problem later?
