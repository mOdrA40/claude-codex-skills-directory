# Reliability and Operations (Ansible)

## Operational Defaults

- Monitor execution failures, privilege issues, drift backlog, inventory integrity, and secret retrieval paths.
- Keep core automation paths tested and boring.
- Document safe rerun and emergency-use playbooks explicitly.
- Distinguish automation failure from target-system failure quickly.

## Run-the-System Thinking

- Critical automation should have clear ownership and support paths.
- Scale increases the importance of staged execution and output clarity.
- Shared automation becomes platform infrastructure once enough teams rely on it.
- Recovery from bad automation should be rehearsed, not improvised.

## Principal Review Lens

- Which automation workflow is most fragile under stress?
- Can the team recover from a bad playbook run quickly?
- What operational practice would most increase trust in Ansible?
- Are we operating a disciplined automation system or a YAML shell farm?
