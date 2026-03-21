# Release Orchestration and Deployment Safety

## Rules

- Release workflows should prioritize predictable promotion, approval, and rollback posture.
- Build, test, package, and deploy steps should expose trustworthy evidence to reviewers.
- Artifact integrity and environment separation are central to release safety.
- One-click release convenience should never hide high-blast-radius behavior.

## Practical Guidance

- Keep release stages explicit and observable.
- Separate validation from deployment privilege where useful.
- Preserve provenance and deployment metadata.
- Make rollback or stop-the-line actions operationally obvious.

## Principal Review Lens

- Can the team explain exactly what a release workflow deploys and where?
- Which deployment step is syntactically clean but operationally dangerous?
- What release control most reduces bad deploy risk?
- Are we automating safety or automating overconfidence?
