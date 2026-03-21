# State, Secrets, and Backend Safety

## Rules

- State is operational truth and must be protected accordingly.
- Secret handling must reflect both application and platform trust boundaries.
- Backend availability and access policy affect deployment safety materially.
- Stack recovery and import workflows should be documented before crisis.

## Practical Guidance

- Separate secret exposure concerns in code, config, CI, and state backend.
- Test backup and restore of state where infra criticality is high.
- Know how imports, moves, and refactors affect stack safety.
- Keep backend choice aligned with team size and compliance needs.

## Principal Review Lens

- Which stack has the weakest secret posture today?
- What happens if the backend is unavailable during a critical change?
- Can the team recover from state corruption or bad import safely?
- Are we honest about where sensitive truth is stored?
