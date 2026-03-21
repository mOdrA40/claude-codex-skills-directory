# Incident Runbooks (Ansible)

## Cover at Minimum

- Bad automation rollout.
- Inventory or variable precedence mistake.
- Privilege or secret access failure.
- Partial execution on a critical fleet.
- Emergency rerun or rollback workflow.
- Shared role regression.

## Response Rules

- Stabilize target systems before cleaning automation structure.
- Preserve logs, diffs, and variable context for RCA.
- Prefer targeted remediation over immediate full reruns.
- Communicate clearly when automation is unsafe until reconciled.

## Principal Review Lens

- Can responders stop blast radius quickly?
- Which emergency action risks widening configuration drift?
- What evidence proves the fleet is actually reconciled again?
- Are runbooks practical for real operator mistakes?
