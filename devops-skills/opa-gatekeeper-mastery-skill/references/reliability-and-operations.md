# Reliability and Operations (OPA Gatekeeper)

## Operational Defaults

- Monitor webhook health, deny rates, admission latency, audit backlog, and template/constraint errors.
- Keep policy rollouts staged and reversible.
- Distinguish cluster policy outages from app config mistakes quickly.
- Document safe disable, bypass, or rollback workflows for severe incidents.

## Run-the-System Thinking

- Policy platforms deserve reliability standards when they sit in the admission path.
- Shared templates and global constraints carry high blast radius.
- Capacity planning should include admission volume and audit scope growth.
- Operational trust comes from explainable policy and controlled enforcement.

## Principal Review Lens

- Which Gatekeeper failure blocks the most teams fastest?
- Can the team stop a policy-induced outage safely?
- What operational habit most improves policy platform trust?
- Are we enforcing valuable controls or fragile bureaucracy?
