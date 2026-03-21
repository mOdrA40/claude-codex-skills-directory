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

## Schema Heuristics

### Mapping choices are long-tail commitments

Every important field choice affects storage, query behavior, relevance, reindex cost, and incident recovery complexity later.

### Analyzer behavior is product behavior

Tokenization, stemming, normalization, and keyword strategy shape what users perceive as search quality. These are not merely low-level text settings.

### Dynamic convenience must stay bounded

Dynamic mappings can accelerate onboarding, but without review they often create field sprawl, inconsistent types, and expensive cleanup later.

## Common Failure Modes

### Schema convenience debt

The team moves fast early with dynamic or loosely governed templates, then pays heavily when relevance, storage, or reindex safety matters.

### Analyzer change without product review

Search behavior shifts materially, but teams discuss it as a technical tweak rather than a user-visible product change.

## Principal Review Lens

- Which field is most likely to become costly to fix later?
- Are analyzers aligned with real user language and search expectations?
- What mapping drift is already creating hidden tax?
- Can the team reindex safely when schema must change?
- Which mapping decision is easiest today but most dangerous over the next year?
