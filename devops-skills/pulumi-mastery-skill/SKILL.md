---
name: pulumi-principal-engineer
description: |
  Principal/Senior-level Pulumi playbook for programmatic IaC architecture, stack design, policy, state safety, multi-language tradeoffs, and production infrastructure delivery.
  Use when: designing Pulumi platforms, reviewing stack organization, operating shared IaC workflows, or balancing code power against infra safety.
---

# Pulumi Mastery (Senior → Principal)

## Operate

- Start from ownership boundaries, stack isolation, and change safety.
- Treat Pulumi as software-defined infrastructure with all the risks of both software and infra.
- Prefer clear stack contracts and predictable deployment behavior over language cleverness.
- Optimize for reviewability, safe rollout, and recovery from bad code or bad state.

## Default Standards

- Stack boundaries should reflect ownership and blast radius.
- Program logic must stay understandable by infrastructure reviewers.
- State and secrets posture must be explicit.
- Policy and guardrails should constrain dangerous flexibility.
- Language power should not justify hidden infrastructure behavior.

## References

- Stack design and ownership boundaries: [references/stack-design-and-ownership-boundaries.md](references/stack-design-and-ownership-boundaries.md)
- Program architecture and abstraction discipline: [references/program-architecture-and-abstraction-discipline.md](references/program-architecture-and-abstraction-discipline.md)
- State, secrets, and backend safety: [references/state-secrets-and-backend-safety.md](references/state-secrets-and-backend-safety.md)
- Reviewability and deployment safety: [references/reviewability-and-deployment-safety.md](references/reviewability-and-deployment-safety.md)
- Policy, governance, and multi-team platforms: [references/policy-governance-and-multi-team-platforms.md](references/policy-governance-and-multi-team-platforms.md)
- Multi-language and platform tradeoffs: [references/multi-language-and-platform-tradeoffs.md](references/multi-language-and-platform-tradeoffs.md)
- Reliability and operations: [references/reliability-and-operations.md](references/reliability-and-operations.md)
- Incident runbooks: [references/incident-runbooks.md](references/incident-runbooks.md)
