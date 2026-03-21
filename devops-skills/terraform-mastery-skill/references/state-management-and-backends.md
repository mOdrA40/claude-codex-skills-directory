# State Management and Backends

## Rules

- State is shared operational truth and must be treated as critical infrastructure.
- Backend choice should reflect concurrency, durability, access control, and disaster recovery requirements.
- State boundaries should minimize team interference and failure blast radius.
- Imports, moves, and refactors must preserve state integrity.

## Operational Thinking

- Locking behavior matters when teams or automation scale.
- Backup and restore of state must be tested.
- Sensitive values in state require explicit handling and trust boundaries.
- One giant state file is often convenience debt.

## Principal Review Lens

- Which state file is currently too large or too shared?
- What happens if the backend is unavailable during a critical change window?
- Can the team recover from accidental state corruption quickly?
- Are we using state layout to encode ownership or just historical accidents?
