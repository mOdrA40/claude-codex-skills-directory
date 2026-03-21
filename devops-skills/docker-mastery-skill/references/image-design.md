# Image Design (Docker)

## Rules

- Images should be minimal, reproducible, and reviewable.
- Every package increases attack surface and upgrade burden.
- Keep runtime images separate from build tooling.
- Design image contents around actual runtime needs.

## Principal Review Lens

- What is inside this image that does not need to be there?
- Is image size hiding bigger supply-chain or patching issues?
- Can operators understand what actually runs in production?
