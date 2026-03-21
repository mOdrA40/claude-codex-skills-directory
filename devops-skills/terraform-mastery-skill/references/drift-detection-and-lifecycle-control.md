# Drift Detection and Lifecycle Control

## Rules

- Drift is a governance and incident risk, not only an aesthetic mismatch.
- Teams need a clear policy for manual changes, emergency fixes, and reconciliation.
- Lifecycle meta-arguments should be used carefully and with explicit intent.
- Import and state surgery workflows should be documented before they are needed urgently.

## Failure Modes

- Emergency console changes becoming permanent hidden truth.
- Ignore rules masking real resource divergence.
- Drift detected but no owner or action path exists.
- Overuse of replacement-inducing changes without rollout awareness.

## Principal Review Lens

- Which resources are most likely to drift silently?
- Are we using ignore rules to preserve safety or to hide debt?
- Can the platform detect dangerous divergence early?
- What manual fix today will confuse tomorrow's apply the most?
