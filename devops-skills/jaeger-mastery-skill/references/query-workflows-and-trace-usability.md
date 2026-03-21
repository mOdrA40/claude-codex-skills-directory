# Query Workflows and Trace Usability

## Rules

- Trace query workflows should reflect how responders actually investigate systems.
- Search dimensions must be governed enough to be useful.
- Trace UI and query habits should connect to common incident entry points.
- Usability matters as much as ingest volume.

## Practical Guidance

- Standardize the service, operation, and deployment dimensions people rely on.
- Teach teams which workflows are reliable and which are best-effort.
- Keep trace lookups fast enough for real incident use.
- Surface known gaps caused by sampling or storage limits.

## Principal Review Lens

- Can responders find useful traces quickly?
- Which missing dimension most hurts search quality?
- Are users expecting capabilities the current setup cannot deliver?
- What usability improvement most reduces incident time?
