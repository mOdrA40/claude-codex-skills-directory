# Multi-Tenant Governance and Policy Ownership

## Rules

- Shared clusters need clear ownership of policy intent, rollout, and exception paths.
- App teams should understand the policies affecting them.
- Tenant autonomy should not undermine cluster safety.
- Policy ownership should be visible enough to support incidents and platform evolution.

## Practical Guidance

- Define who owns global policies, namespace-scoped policies, and exceptions.
- Standardize review workflows for high-impact constraints.
- Communicate policy changes with enough lead time and rationale.
- Keep project or tenant-specific carveouts understandable and minimal.

## Principal Review Lens

- Which team can create the biggest policy blast radius today?
- Are ownership lines clear when policy blocks production?
- What communication gap is causing repeated friction?
- Which governance change most improves platform trust?
