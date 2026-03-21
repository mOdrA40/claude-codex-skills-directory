# Reliability and Operations (Argo CD)

## Operational Defaults

- Monitor reconciliation health, repo access, render failures, controller status, and cluster apply errors.
- Keep upgrades and policy changes staged and reversible.
- Distinguish global platform issues from one-app problems quickly.
- Document safe emergency stop, rollback, and reconciliation workflows.

## Run-the-System Thinking

- GitOps control planes deserve SLO thinking when many teams depend on them.
- Shared configuration and template changes can have very high blast radius.
- Capacity planning should include repo churn, app count, and render complexity.
- Operational trust comes from predictable reconciliation and clear governance.

## Principal Review Lens

- Which Argo failure mode blocks the most teams fastest?
- Can the team safely pause or recover delivery during crisis?
- What practice most improves GitOps platform trust?
- Are we running declarative delivery or declarative confusion?
