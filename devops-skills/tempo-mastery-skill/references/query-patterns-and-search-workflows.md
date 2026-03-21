# Query Patterns and Search Workflows

## Rules

- Trace search workflows should match how responders investigate incidents.
- Query posture must reflect what metadata exists and how reliable it is.
- Broad trace search can become expensive and disappointingly weak if instrumentation is inconsistent.
- Tempo UX should be aligned with correlation entry points from dashboards and logs.

## Design Guidance

- Standardize the fields operators use to pivot into traces.
- Make search limitations explicit where metadata or retention is thin.
- Avoid promising magical search across poorly governed telemetry.
- Teach teams how to move from symptom to representative traces efficiently.

## Principal Review Lens

- Can responders find the right traces quickly today?
- Which missing metadata most hurts search usefulness?
- Are we overtrusting trace search where instrumentation is weak?
- What search workflow deserves productization next?
