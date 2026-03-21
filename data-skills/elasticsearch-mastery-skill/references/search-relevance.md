# Search Relevance (Elasticsearch)

## Rules

- Relevance requires labeled query sets and evaluation discipline.
- Boosting without measurement is cargo cult.
- Synonyms and ranking rules should be versioned and reviewed.
- Distinguish business ranking from textual similarity.

## Relevance Heuristics

### Search quality should be measured where it matters most

The most important evaluation set is not generic search traffic. It is the subset of queries that most affects user trust, conversion, task success, or revenue.

### Ranking rules are product policy

Boosts, synonyms, merchandising, and business ranking encode explicit product choices and should be reviewed with that seriousness.

### Relevance work must remain debuggable

If the team cannot explain why a result ranks where it does, tuning becomes guesswork and incident handling becomes much slower.

## Common Failure Modes

### Boost cargo cult

The team stacks boosts and synonyms faster than it improves evaluation discipline, making search behavior less explainable over time.

### Business ranking hidden in search plumbing

Commercial or policy choices creep into low-level scoring without enough visibility or versioning.

### Query-set illusion

Relevance is tuned on a narrow or outdated sample that does not reflect the user journeys that actually matter most.

## Principal Review Lens

- Which queries matter most to users or revenue?
- What relevance tradeoff is accepted explicitly?
- Can the team validate a ranking change safely?
- Which ranking rule is most weakly justified by actual evidence today?
