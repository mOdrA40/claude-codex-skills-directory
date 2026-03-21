# Incident Runbooks (Terraform)

## Cover at Minimum

- State lock stuck or backend outage.
- Bad apply with destructive impact.
- Provider regression.
- Drift-related outage.
- Emergency import or state move.
- Pipeline breakage blocking infrastructure recovery.

## Response Rules

- Stabilize production behavior before cleaning up IaC aesthetics.
- Protect state integrity during emergency action.
- Record all manual interventions for later reconciliation.
- Prefer reversible mitigation before large refactors in crisis.

## Principal Review Lens

- Can responders recover service without making state worse?
- Which emergency action most risks long-term infrastructure confusion?
- What evidence is required before re-running apply?
- Are runbooks specific enough for someone other than the module author?
