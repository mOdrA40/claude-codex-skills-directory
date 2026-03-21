# Multi-Repo Governance and Platform Standards

## Rules

- At scale, CI/CD quality depends on shared standards, ownership, and lifecycle management.
- Platform teams should publish secure defaults and supportable golden paths.
- Exceptions should be explicit and reviewable.
- Governance should reduce repeated failure patterns, not only document them.

## Practical Guidance

- Standardize workflows for build, test, release, dependency updates, and security scanning where useful.
- Track ownerless or stale workflows.
- Version platform automation contracts deliberately.
- Measure adoption and friction to avoid governance theater.

## Principal Review Lens

- Which repo pattern is producing most CI/CD drift?
- Are teams bypassing the platform because the golden path is weak?
- What ownerless automation is most dangerous today?
- Which standard would create the highest leverage if enforced next?
