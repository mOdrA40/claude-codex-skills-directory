# Pipeline Architecture and Task Boundaries (Tekton)

## Rules

- Pipelines should reflect delivery stages and ownership boundaries clearly.
- Tasks should encapsulate coherent units of work, not giant all-in-one shells.
- Reuse should improve consistency without hiding build semantics.
- Pipeline structure should remain operable under failure.

## Design Guidance

- Separate build, test, scan, package, and deploy concerns where blast radius differs.
- Keep task interfaces explicit and narrow.
- Make artifact flow and handoff points obvious.
- Avoid composition patterns that make rerun and debugging hard.

## Principal Review Lens

- Which task boundary is too broad for safe reuse?
- Can reviewers explain pipeline flow without reading every script?
- What composition choice most increases debugging pain?
- Which pipeline split would improve safety most?
