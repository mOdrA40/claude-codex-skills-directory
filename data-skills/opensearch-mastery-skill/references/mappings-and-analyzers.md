# Mappings and Analyzers (OpenSearch)

## Rules

- Explicit mappings on important fields reduce future operational pain.
- Analyzer choice is part of product behavior, not just text plumbing.
- Separate exact-match, keyword, and full-text needs intentionally.
- Dynamic mapping convenience should be bounded and reviewed.

## Design Guidance

- Standardize field strategy for timestamps, identifiers, text, enums, and numerics.
- Version analyzer changes and prepare for reindex impact.
- Avoid mapping explosion from uncontrolled ingest sources.
- Keep index template ownership clear.

## Principal Review Lens

- Which field is most likely to become costly to fix later?
- Are analyzers aligned with real user language and search expectations?
- What mapping drift is already creating hidden tax?
- Can the team reindex safely when schema must change?
