# Reliability and Operations (GitHub Actions)

## Operational Defaults

- Monitor workflow failure rates, queueing, runner health, secret/access errors, and flaky job patterns.
- Keep shared automation changes staged and reversible.
- Distinguish GitHub platform issues from self-hosted runner or workflow logic issues quickly.
- Document emergency release and rollback workflows outside of fragile assumptions.

## Run-the-System Thinking

- CI/CD platforms deserve reliability standards if many teams depend on them.
- Shared automation becomes platform infrastructure once the organization standardizes on it.
- Incident handling should preserve evidence from jobs, artifacts, and environment gates.
- Boring, supportable workflows beat clever YAML.

## Principal Review Lens

- Which failure mode blocks the most teams fastest?
- Can the team deploy safely when parts of automation are degraded?
- What operational practice most increases platform trust?
- Are we operating CI/CD as infrastructure or as repo decoration?
