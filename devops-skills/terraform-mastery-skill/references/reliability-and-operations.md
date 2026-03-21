# Reliability and Operations (Terraform)

## Operational Defaults

- Monitor pipeline failures, backend health, lock contention, provider breakage, and drift backlog.
- Keep version upgrades staged and reversible.
- Backup state and rehearse recovery for foundational stacks.
- Treat IaC delivery outages as platform incidents when many teams depend on them.

## Run-the-System Thinking

- Document who can approve, apply, override, and recover changes.
- Foundational stacks deserve higher rigor than ephemeral experimentation stacks.
- Provider/API instability should influence rollout posture.
- Support workflows for imports and drift repair should not require heroes.

## Principal Review Lens

- What infrastructure change workflow is most fragile today?
- Which backend or pipeline issue could block critical recovery work?
- Can the team recover from bad apply, bad provider, or bad state fast enough?
- Are we operating Terraform as a platform or as a collection of scripts?
