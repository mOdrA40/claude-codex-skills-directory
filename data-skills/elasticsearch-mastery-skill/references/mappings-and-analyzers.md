# Mappings and Analyzers (Elasticsearch)

## Rules

- Important fields deserve explicit mappings.
- Analyzer choice is product behavior, not only text plumbing.
- Separate exact-match and full-text needs intentionally.
- Reindex cost should influence mapping ambition.

## Principal Review Lens

- Which field is over-flexible and will hurt later?
- Do analyzers match user language and query behavior?
- What mapping change would force painful reindexing?
