# Incident Runbooks (CockroachDB)

## Rules

- Runbooks should cover contention storms, regional impairment, node loss, and latency regressions.
- Start with blast-radius reduction before perfect diagnosis.
- Include safe operator actions and forbidden shortcuts.
- Tie recovery steps to measurable signals.

## Principal Review Lens

- Can on-call stabilize retry storms quickly?
- Which action risks making placement or availability worse?
- What metric proves real recovery rather than temporary calm?
