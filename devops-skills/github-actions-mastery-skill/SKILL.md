---
name: github-actions-principal-engineer
description: |
  Principal/Senior-level GitHub Actions playbook for CI/CD architecture, workflow design, runner strategy, supply-chain safety, release governance, and operating automation platforms at scale.
  Use when: designing pipelines, reviewing workflow security, scaling runners, hardening delivery automation, or governing repository automation across many teams.
---

# GitHub Actions Mastery (Senior → Principal)

## Operate

- Start from change safety, trust boundaries, and developer feedback loops.
- Treat GitHub Actions as production automation infrastructure, not just YAML attached to repos.
- Prefer explicit workflow boundaries, reusable patterns, and reviewable release paths.
- Optimize for safe automation, fast feedback, and low blast radius under failure.

## Default Standards

- Workflow triggers must be intentional.
- Secrets, tokens, and permissions should be least-privilege.
- Reusable workflows should reduce duplication without hiding critical behavior.
- Runner strategy should reflect workload isolation and security posture.
- Supply-chain and artifact trust must be explicit.

## References

- Workflow architecture and event boundaries: [references/workflow-architecture-and-event-boundaries.md](references/workflow-architecture-and-event-boundaries.md)
- Reusable workflows and action governance: [references/reusable-workflows-and-action-governance.md](references/reusable-workflows-and-action-governance.md)
- Runner strategy and isolation: [references/runner-strategy-and-isolation.md](references/runner-strategy-and-isolation.md)
- Secrets, permissions, and supply-chain safety: [references/secrets-permissions-and-supply-chain-safety.md](references/secrets-permissions-and-supply-chain-safety.md)
- Release orchestration and deployment safety: [references/release-orchestration-and-deployment-safety.md](references/release-orchestration-and-deployment-safety.md)
- Multi-repo governance and platform standards: [references/multi-repo-governance-and-platform-standards.md](references/multi-repo-governance-and-platform-standards.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
