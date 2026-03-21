# Incident Runbooks (Helm)

## Cover at Minimum

- Bad chart rollout.
- Failed rollback.
- Hook/job failures.
- Registry or dependency unavailability.
- Values misconfiguration causing broken manifests.
- Shared chart upgrade regression.

## Response Rules

- Restore workload health before cleaning chart elegance.
- Preserve rendered manifests and release metadata for RCA.
- Prefer proven rollback paths over speculative manual fixes.
- Communicate when the issue is chart packaging versus Kubernetes runtime.

## Principal Review Lens

- Can operators revert safely without the original chart author?
- Which emergency action risks making future releases less trustworthy?
- What evidence should be captured before retrying an upgrade?
- Are runbooks written for realistic failure sequencing?
