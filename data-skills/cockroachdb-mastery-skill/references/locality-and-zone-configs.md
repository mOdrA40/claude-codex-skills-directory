# Locality and Zone Configs (CockroachDB)

## Rules

- Locality settings should express latency and survivability intent.
- Keep placement logic understandable by humans on-call.
- Avoid accidental cross-region tax on hot paths.
- Validate configuration against actual client geography.

## Principal Review Lens

- Which requests cross regions unnecessarily?
- Does placement match business criticality and data sovereignty needs?
- Can operators predict where data and leaseholders live?
