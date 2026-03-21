# Security, mTLS, and Identity

## Rules

- Workload identity is a security boundary and must be governed accordingly.
- mTLS posture should reflect real trust requirements, not checkbox security.
- Authorization policy should remain understandable and auditable.
- Certificate issuance and rotation are operational reliability concerns too.

## Design Guidance

- Define where strict, permissive, or disabled modes are acceptable.
- Make service identity naming and namespace boundaries meaningful.
- Keep authz policy close to ownership and review workflows.
- Test policy behavior during migrations and mixed-mode states.

## Principal Review Lens

- Which workload has the weakest effective identity boundary?
- Can policy changes accidentally cut off critical traffic?
- Are operators confident enough to debug mTLS failures fast?
- What security control is present in YAML but weak in reality?
