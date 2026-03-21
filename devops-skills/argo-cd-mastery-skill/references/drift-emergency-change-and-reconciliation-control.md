# Drift, Emergency Change, and Reconciliation Control

## Rules

- Drift is a delivery and incident-management problem, not just a UI status.
- Emergency cluster-side changes need explicit reconciliation policy.
- Operators should know when to pause, force, or ignore reconciliation safely.
- Desired state and live state disagreements must have owners and workflows.

## Practical Guidance

- Document emergency break-glass procedures and follow-up reconciliation steps.
- Avoid silent drift tolerance that hides long-term danger.
- Use sync options and ignores carefully, not as permanent debt dumping grounds.
- Keep change history and rationale visible.

## Principal Review Lens

- Which drift pattern is most likely to become a future incident?
- Are we using ignore settings to preserve safety or to hide poor design?
- Can responders stop Argo from making a crisis worse?
- What workflow most improves trust after emergency manual changes?
