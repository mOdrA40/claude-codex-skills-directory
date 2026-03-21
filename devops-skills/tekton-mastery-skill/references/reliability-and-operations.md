# Reliability and Operations (Tekton)

## Operational Defaults

- Monitor controller health, task execution failures, queueing, workspace/storage issues, and cluster dependency behavior.
- Keep pipeline platform upgrades staged and reversible.
- Distinguish shared platform incidents from single-team pipeline failures quickly.
- Document safe fallback release paths when the platform degrades.

## Run-the-System Thinking

- Tekton platforms deserve reliability standards when many teams depend on them.
- Capacity planning includes runner resources, cluster scheduling, registry throughput, and artifact storage.
- Operational trust comes from explicit contracts and boring task design.
- Shared task and bundle changes can carry high blast radius.

## Principal Review Lens

- Which Tekton failure blocks the most teams fastest?
- Can the team preserve safe delivery during partial platform outage?
- What operational habit most improves platform trust?
- Are we running a disciplined pipeline platform or YAML-shaped entropy?
