# Secrets, Privilege, and Security Posture

## Rules

- Secrets handling and privilege escalation should be explicit, minimal, and auditable.
- Vaulting or secret retrieval mechanisms must reflect trust boundaries.
- Become/use of elevated permissions should be constrained to what is necessary.
- Automation accounts are part of the attack surface.

## Practical Guidance

- Separate secret distribution from general variable sprawl.
- Review where plaintext exposure can happen in logs, templates, or debug output.
- Keep key rotation and secret update workflows well understood.
- Avoid broad sudo patterns that exceed task needs.

## Principal Review Lens

- Which automation path has more privilege than it should?
- Can secrets leak through a normal troubleshooting workflow?
- Are we relying on convention rather than enforceable security boundaries?
- What change most strengthens automation security posture?
