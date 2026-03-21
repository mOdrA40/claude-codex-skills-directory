# Reliability and Operations (Helm)

## Operational Defaults

- Track release failures, drift between desired and actual manifests, and upgrade pain points.
- Chart repository, OCI registry, and dependency availability are part of operational reliability.
- Keep rollback workflows documented and exercised.
- Separate Helm problems from Kubernetes runtime problems quickly.

## Run-the-System Thinking

- Platform teams should know which releases are highest blast radius.
- Release metadata and history should be usable during incidents.
- Chart deprecation and version cleanup are operational maintenance, not optional chores.
- Major chart changes deserve staged rollout and communication.

## Principal Review Lens

- Which release workflow is most likely to fail under pressure?
- Can operators determine whether failure is chart logic or cluster behavior fast?
- What dependency outage could block critical rollouts?
- Are we operating Helm as a trustworthy release layer or as a YAML disguise?
