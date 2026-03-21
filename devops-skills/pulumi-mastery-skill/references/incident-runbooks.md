# Incident Runbooks (Pulumi)

## Cover at Minimum

- Bad deployment rollout.
- Backend or state failure.
- Secret access or config issue.
- Preview mismatch versus actual impact.
- Provider/API regression.
- Emergency import or state repair.

## Response Rules

- Stabilize infrastructure behavior before cleaning code elegance.
- Preserve deployment previews, logs, and state evidence for RCA.
- Prefer targeted recovery over broad redeploy panic.
- Communicate clearly when stack state is temporarily untrusted.

## Principal Review Lens

- Can responders stop blast radius quickly?
- Which emergency action most risks worsening state integrity?
- What evidence proves the stack is reconciled again?
- Are runbooks good enough for someone other than the original author?
