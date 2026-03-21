# Reliability and Operations (Pulumi)

## Operational Defaults

- Monitor deployment failures, backend health, state integrity, preview drift, and secret access issues.
- Keep core stacks and deployment workflows boring and tested.
- Distinguish program bugs, provider/API instability, and state issues quickly.
- Document emergency recovery paths for bad deployments.

## Run-the-System Thinking

- Shared Pulumi platforms become critical infrastructure once many teams depend on them.
- Capacity is often about human review and deployment safety as much as runtime performance.
- Stack recovery and rollback workflows should not depend on heroes.
- Policy and golden-path quality determine whether scale creates chaos or leverage.

## Principal Review Lens

- Which deployment workflow is most fragile today?
- Can the team recover from a bad stack update quickly?
- What operational habit most improves Pulumi trust?
- Are we operating disciplined programmable IaC or just infra code with weak guardrails?
